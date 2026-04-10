from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

try:
    import umap
except Exception:  # pragma: no cover - optional dependency runtime check
    umap = None


@dataclass
class ClusteringError(Exception):
    code: str
    message: str
    status_code: int = 400

    def to_response(self) -> Dict[str, Dict[str, str]]:
        return {"error": {"code": self.code, "message": self.message}}


def _ensure_finite_number(value: Any) -> float:
    if not isinstance(value, (int, float, np.floating, np.integer)):
        raise ClusteringError("INVALID_FEATURE_VALUE", "All feature values must be numeric.", 422)
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ClusteringError("INVALID_FEATURE_VALUE", "All feature values must be finite numbers.", 422)
    return numeric


def _build_feature_matrix(samples: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
    if not samples:
        raise ClusteringError("EMPTY_SAMPLES", "Samples list must not be empty.", 422)

    normalized_samples: List[Dict[str, Any]] = []
    expected_keys: List[str] | None = None

    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ClusteringError("INVALID_SAMPLE", f"Sample at index {index} must be an object.", 422)

        sample_id = sample.get("sample_id")
        label = sample.get("label")
        features = sample.get("features")

        if not isinstance(sample_id, int) or sample_id <= 0:
            raise ClusteringError("INVALID_SAMPLE_ID", f"Invalid sample_id at index {index}.", 422)
        if not isinstance(label, str) or label.strip() == "":
            raise ClusteringError("INVALID_LABEL", f"Invalid label at index {index}.", 422)
        if not isinstance(features, dict) or not features:
            raise ClusteringError("EMPTY_FEATURES", f"Missing features for sample {sample_id}.", 422)

        keys = sorted(features.keys())
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise ClusteringError(
                "FEATURE_SCHEMA_MISMATCH",
                "All samples must have identical feature keys in one clustering run.",
                422,
            )

        normalized_features = {key: _ensure_finite_number(features[key]) for key in keys}
        normalized_samples.append(
            {
                "sample_id": sample_id,
                "label": label,
                "features": normalized_features,
                "meta": sample.get("meta") if isinstance(sample.get("meta"), dict) else None,
            }
        )

    if expected_keys is None or len(expected_keys) < 2:
        raise ClusteringError("INSUFFICIENT_FEATURES", "At least 2 features are required.", 422)
    if len(normalized_samples) < 3:
        raise ClusteringError("INSUFFICIENT_SAMPLES", "At least 3 samples are required.", 422)

    matrix = np.array(
        [[sample["features"][feature_key] for feature_key in expected_keys] for sample in normalized_samples],
        dtype=float,
    )

    return matrix, expected_keys, normalized_samples


def _build_embedding(matrix: np.ndarray, method: str, random_state: int) -> Tuple[np.ndarray, List[Dict[str, str]]]:
    warnings: List[Dict[str, str]] = []

    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        embedding = reducer.fit_transform(matrix)
        return embedding, warnings

    if method == "umap":
        if umap is None:
            raise ClusteringError("UMAP_UNAVAILABLE", "UMAP is not installed on the server.", 500)

        n_samples = int(matrix.shape[0])
        if n_samples < 4:
            # UMAP spectral init is unstable for very small N; degrade gracefully.
            reducer = PCA(n_components=2, random_state=random_state)
            embedding = reducer.fit_transform(matrix)
            warnings.append(
                {
                    "code": "UMAP_FALLBACK_PCA",
                    "message": "Too few samples for stable UMAP; PCA embedding was used instead.",
                }
            )
            return embedding, warnings

        n_neighbors = min(15, max(2, matrix.shape[0] - 1))
        reducer = umap.UMAP(
            n_components=2,
            random_state=random_state,
            n_neighbors=n_neighbors,
            init="random",
        )
        try:
            embedding = reducer.fit_transform(matrix)
        except Exception:
            reducer = PCA(n_components=2, random_state=random_state)
            embedding = reducer.fit_transform(matrix)
            warnings.append(
                {
                    "code": "UMAP_FALLBACK_PCA",
                    "message": "UMAP failed on current sample set; PCA embedding was used instead.",
                }
            )
            return embedding, warnings

        warnings.append(
            {
                "code": "UMAP_STOCHASTIC",
                "message": "UMAP may produce less stable embeddings than PCA.",
            }
        )
        return embedding, warnings

    raise ClusteringError("UNSUPPORTED_EMBEDDING_METHOD", f"Unsupported embedding method: {method}", 422)


def _cluster(matrix: np.ndarray, clusters_count: int, random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if clusters_count >= matrix.shape[0]:
        raise ClusteringError("INVALID_CLUSTERS_COUNT", "clusters_count must be less than sample count.", 422)

    model = KMeans(n_clusters=clusters_count, random_state=random_state, n_init=10)
    labels = model.fit_predict(matrix)
    centers = model.cluster_centers_

    distances = pairwise_distances(matrix, centers)
    min_distances = np.min(distances, axis=1)

    return labels, centers, min_distances


def _build_cluster_summary(labels: np.ndarray) -> List[Dict[str, int]]:
    unique_labels, counts = np.unique(labels, return_counts=True)
    return [
        {"cluster": int(cluster_label), "size": int(cluster_count)}
        for cluster_label, cluster_count in zip(unique_labels, counts)
    ]


def _build_warnings(sample_count: int, clusters_count: int, cluster_summary: List[Dict[str, int]]) -> List[Dict[str, str]]:
    warnings: List[Dict[str, str]] = []

    if sample_count < 5:
        warnings.append(
            {
                "code": "LOW_SAMPLE_COUNT",
                "message": "Sample count is low; clustering may be unstable.",
            }
        )

    if clusters_count >= max(2, sample_count - 1):
        warnings.append(
            {
                "code": "K_NEAR_N",
                "message": "clusters_count is very close to number of samples.",
            }
        )

    if cluster_summary:
        sizes = [cluster_item["size"] for cluster_item in cluster_summary]
        if min(sizes) <= 1 or (max(sizes) / max(min(sizes), 1)) >= 6:
            warnings.append(
                {
                    "code": "UNBALANCED_CLUSTERS",
                    "message": "Detected strong cluster size imbalance.",
                }
            )

    return warnings


def build_clustering_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    matrix, feature_keys, normalized_samples = _build_feature_matrix(payload.get("samples", []))

    normalize = bool(payload.get("normalize", True))
    embedding_method = payload.get("embedding_method", "pca")
    clustering_method = payload.get("clustering_method", "kmeans")
    clusters_count = int(payload.get("clusters_count", 4))
    random_state = int(payload.get("random_state", 42))
    return_debug = bool(payload.get("return_debug", False))

    if clustering_method != "kmeans":
        raise ClusteringError("UNSUPPORTED_CLUSTERING_METHOD", f"Unsupported clustering method: {clustering_method}", 422)

    matrix_for_model = matrix
    scaler = None
    if normalize:
        scaler = StandardScaler()
        matrix_for_model = scaler.fit_transform(matrix)

    embedding, embedding_warnings = _build_embedding(matrix_for_model, embedding_method, random_state)
    labels, centers, min_distances = _cluster(matrix_for_model, clusters_count, random_state)

    embedding_rows: List[Dict[str, Any]] = []
    for idx, sample in enumerate(normalized_samples):
        embedding_rows.append(
            {
                "sample_id": int(sample["sample_id"]),
                "label": sample["label"],
                "x": float(embedding[idx, 0]),
                "y": float(embedding[idx, 1]),
                "cluster": int(labels[idx]),
                "distance_to_cluster_center": float(min_distances[idx]),
                "meta": sample.get("meta") or None,
            }
        )

    cluster_summary = _build_cluster_summary(labels)
    warnings = _build_warnings(len(normalized_samples), clusters_count, cluster_summary)
    warnings.extend(embedding_warnings)

    response: Dict[str, Any] = {
        "embedding": embedding_rows,
        "clusters": cluster_summary,
        "summary": {
            "sample_count": len(normalized_samples),
            "feature_count": len(feature_keys),
            "embedding_method": embedding_method,
            "clustering_method": clustering_method,
        },
        "warnings": warnings,
    }

    if return_debug:
        response["debug"] = {
            "feature_keys": feature_keys,
            "normalized": normalize,
            "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "cluster_centers": centers.tolist(),
            "scaler_mean": scaler.mean_.tolist() if scaler is not None else None,
            "scaler_scale": scaler.scale_.tolist() if scaler is not None else None,
        }

    return response
