import numpy as np


class KMeans:

    def __init__(
        self,
        n_clusters=5,
        max_iter=300,
        tol=0.0001,
        random_state=42,
        init="k-means++"
    ):

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None
        self.centroid_history_ = []

    # =================================
    # 1. VALIDATE DATA
    # =================================

    def _validate_data(self, X):

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[0] < self.n_clusters:
            raise ValueError(
                "Number of samples must be >= n_clusters."
            )

        if X.shape[1] == 0 or not np.isfinite(X).all():
            raise ValueError(
                "X must contain valid finite features."
            )

        if self.n_clusters < 1:
            raise ValueError("n_clusters must be >= 1.")

        return X

    # =================================
    # 2. INITIALIZE CENTROIDS
    # =================================

    def _initialize_centroids(self, X):

        rng = np.random.default_rng(self.random_state)

        n_samples = X.shape[0]

        if self.init == "random":

            indices = rng.choice(
                n_samples,
                self.n_clusters,
                replace=False
            )

            return X[indices].copy()

        if self.init != "k-means++":
            raise ValueError(
                "init must be 'random' or 'k-means++'."
            )

        chosen = [rng.integers(n_samples)]

        closest_sq = np.sum(
            (X - X[chosen[0]]) ** 2,
            axis=1
        )

        while len(chosen) < self.n_clusters:

            total = closest_sq.sum()

            if total <= 0:
                remaining = np.setdiff1d(
                    np.arange(n_samples),
                    chosen
                )
                next_idx = rng.choice(remaining)

            else:

                probabilities = closest_sq / total

                next_idx = rng.choice(
                    n_samples,
                    p=probabilities
                )

            if next_idx in chosen:
                remaining = np.setdiff1d(
                    np.arange(n_samples),
                    chosen
                )
                next_idx = rng.choice(remaining)

            chosen.append(next_idx)

            new_distances = np.sum(
                (X - X[next_idx]) ** 2,
                axis=1
            )

            closest_sq = np.minimum(
                closest_sq,
                new_distances
            )

        return X[chosen].copy()

    # =================================
    # 3. CALCULATE DISTANCES
    # =================================

    def _calculate_distances(self, X, centroids):

        distances = np.sum(
            (X[:, np.newaxis, :] -
             centroids[np.newaxis, :, :]) ** 2,
            axis=2
        )

        return distances

    # =================================
    # 4. UPDATE CENTROIDS
    # =================================

    def _update_centroids(self, X, labels, distances):

        new_centroids = np.zeros(
            (self.n_clusters, X.shape[1])
        )

        nearest_distances = distances[
            np.arange(len(X)), labels
        ]

        farthest_indices = np.argsort(
            nearest_distances
        )[::-1]

        empty_cursor = 0

        for cluster_id in range(self.n_clusters):

            cluster_points = X[
                labels == cluster_id
            ]

            if len(cluster_points) > 0:

                new_centroids[cluster_id] = (
                    cluster_points.mean(axis=0)
                )

            else:

                # Reinitialize an empty cluster
                idx = farthest_indices[empty_cursor]
                empty_cursor += 1

                new_centroids[cluster_id] = X[idx]

        return new_centroids

    # =================================
    # 5. TRAIN K-MEANS
    # =================================

    def fit(self, X):

        X = self._validate_data(X)

        centroids = self._initialize_centroids(X)

        # Initialize centroid history
        self.centroid_history_ = [centroids.copy()]

        previous_labels = None

        for iteration in range(1, self.max_iter + 1):

            distances = self._calculate_distances(
                X,
                centroids
            )

            labels = np.argmin(
                distances,
                axis=1
            )

            new_centroids = self._update_centroids(
                X,
                labels,
                distances
            )

            movement = np.linalg.norm(
                new_centroids - centroids
            )

            centroids = new_centroids

            # Save centroid positions at each iteration
            self.centroid_history_.append(centroids.copy())

            labels_stable = (
                previous_labels is not None
                and np.array_equal(labels, previous_labels)
            )

            if movement <= self.tol or labels_stable:

                break

            previous_labels = labels.copy()

        # Final assignment
        final_distances = self._calculate_distances(
            X,
            centroids
        )

        final_labels = np.argmin(
            final_distances,
            axis=1
        )

        self.cluster_centers_ = centroids
        self.labels_ = final_labels

        self.inertia_ = np.sum(
            final_distances[
                np.arange(len(X)),
                final_labels
            ]
        )

        self.n_iter_ = iteration

        return self

    # =================================
    # 6. PREDICT
    # =================================

    def predict(self, X):

        if self.cluster_centers_ is None:
            raise RuntimeError(
                "Model must be fitted before prediction."
            )

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[0] == 0:
            raise ValueError("Input cannot be empty.")

        if X.shape[1] != self.cluster_centers_.shape[1]:
            raise ValueError(
                "Number of features does not match the trained model."
            )

        if not np.isfinite(X).all():
            raise ValueError("Input contains invalid values.")
  
        distances = self._calculate_distances(
            X,
            self.cluster_centers_
        )

        return np.argmin(distances, axis=1)

        

    # =================================
    # 7. FIT PREDICT
    # =================================

    def fit_predict(self, X):

        self.fit(X)

        return self.labels_