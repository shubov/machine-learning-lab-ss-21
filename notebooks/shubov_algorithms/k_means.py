import numpy as np
import random

random.seed(3)
np.seterr(all='raise')


class KMeans:

    def euclidean_distance(self, a, b):
        a_sq = np.reshape(np.sum(a * a, axis=1), (a.shape[0], 1))
        b_sq = np.reshape(np.sum(b * b, axis=1), (1, b.shape[0]))
        ab = np.dot(a, b.transpose())
        c = -2 * ab + b_sq + a_sq
        try:
            return np.sqrt(c)
        except Exception:
            return c

    def get_initial_centroids(self, x, k):
        """ Picks k random unique points from dataset X. Selected points can be used as intial centroids.

        :param x: (numpy.ndarray) dataset points array, size N:D
        :param k: (int) number of centroids
        :returns: (numpy.ndarray) array of k unique initial centroids, size K:D
        """
        num_samples = x.shape[0]
        sample_pt_idx = random.sample(range(0, num_samples), k)
        centroids = [tuple(x[i]) for i in sample_pt_idx]
        unique_centroids = list(set(centroids))
        return np.array(unique_centroids)

    def compute_clusters(self, x, centroids):
        """ Function finds k centroids and assigns each of the N points of array X to one centroid

        :param x: (numpy.ndarray) array of sample points, size N:D
        :param centroids: (numpy.ndarray) array of centroids, size K:D
        :returns: dict {cluster_number: list_of_points_in_cluster}
        """
        k = centroids.shape[0]
        clusters = {}
        distance_mat = self.euclidean_distance(x, centroids)
        closest_cluster_ids = np.argmin(distance_mat, axis=1)

        for i in range(k):
            clusters[i] = []

        for i, cluster_id in enumerate(closest_cluster_ids):
            clusters[cluster_id].append(x[i])

        return clusters

    def check_convergence(self, previous_centroids, new_centroids, movement_threshold_delta):
        """ Function checks if any of centroids moved more than the MOVEMENT_THRESHOLD_DELTA if not we assume
        the centroids were found

        :param previous_centroids: (numpy.ndarray) array of k old centroids, size K:D
        :param new_centroids: (numpy.ndarray) array of k new centroids, size K:D
        :param movement_threshold_delta: (float) threshold value, if centroids move less we assume
            that algorithm covered

        :returns: (boolean) True if centroids coverd False if not

        """
        distances_between_old_and_new_centroids = self.euclidean_distance(previous_centroids, new_centroids)
        converged = np.max(distances_between_old_and_new_centroids.diagonal()) <= movement_threshold_delta
        return converged

    def do_k_means(self, x, k, movement_threshold_delta=0):
        """Performs k-means algorithm on a given dataset, finds and returns k centroids

        :param x: (numpy.ndarray) dataset points array, size N:D
            between all points from matrix A and all points from matrix B, size N1:N2
        :param k: (int) number of centroids
        :param movement_threshold_delta: (float) threshold value, if centroids move less we assume
            that algorithm covered

        :returns: (numpy.ndarray) array of k centroids, size K:D
        """
        new_centroids = self.get_initial_centroids(x=x, k=k)
        converged = False

        while not converged:
            previous_centroids = new_centroids
            clusters = self.compute_clusters(x, previous_centroids)
            new_centroids = np.array([np.mean(clusters[key], axis=0, dtype=x.dtype) for key in sorted(clusters.keys())])
            converged = self.check_convergence(previous_centroids, new_centroids, movement_threshold_delta)
        return new_centroids
