import numpy as np

def k_means_clustering(
    points: list[tuple[float, ...]],
    k: int,
    initial_centroids: list[tuple[float, ...]],
    max_iterations: int
) -> list[tuple[float, ...]]:

    X = np.array(points, dtype=float)
    centroids = np.array(initial_centroids, dtype=float)

    def assign_clusters(X, centroids):
        clusters = []

        for x in X:
            distances = [np.linalg.norm(x - c) for c in centroids]
            cluster = np.argmin(distances)
            clusters.append(cluster)

        return np.array(clusters)

    def update_centroids(X, clusters, centroids, k):
        new_centroids = []

        for i in range(k):
            cluster_points = X[clusters == i]

            if len(cluster_points) == 0:
                # Keep the old centroid if the cluster is empty
                new_centroids.append(centroids[i])
            else:
                new_centroids.append(cluster_points.mean(axis=0))

        return np.array(new_centroids)

    for _ in range(max_iterations):
        clusters = assign_clusters(X, centroids)

        new_centroids = update_centroids(
            X, clusters, centroids, k
        )

        if np.allclose(centroids, new_centroids):
            centroids = new_centroids
            break

        centroids = new_centroids

    final_centroids = [
        tuple(round(float(value), 4) for value in centroid)
        for centroid in centroids
    ]

    return final_centroids