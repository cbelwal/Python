"""
Clustering analysis for comparing algorithms (2, 3) and baselines (11 PCA, 21 Raw)
Computes optimal clusters using WCSS and silhouette scores using K-Means
"""
import os,sys
# ----------------------------------------------
# Explicit declaration to ensure the root folder path is in sys.path
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(topRootPath)
#----------------------------------------------
from Experiments.Database.CDatabaseManager import CDatabaseManager
from Experiments.ExecuteExperiments.Helpers.CResultsStore import CResultsStore
from Experiments.ExecuteExperiments.Helpers.CDistanceAnalysis_Baseline_Raw import CDistanceAnalysis_Baseline_Raw
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
import numpy as np

MAX_NUMBER_OF_CLUSTERS = 10
DEFAULT_NUMBER_OF_CLUSTERS = 3

class CClusteringAnalysis_Baseline:
    # Algorithm IDs: 2, 3 are main algorithms; 11 is PCA baseline; 21 is raw tool counts
    ALGORITHM_IDS = [2, 3, 11]
    RAW_ALG_ID = 21

    def __init__(self):
        self.dbManager = CDatabaseManager()
        self.all_user_ids = self.dbManager.get_all_user_ids()

        # Load embeddings for algorithms 2, 3, 11
        print("Loading embeddings for algorithms...")
        self.embeddings_by_alg = {}
        for alg_id in self.ALGORITHM_IDS:
            store = CResultsStore(algID=alg_id)
            self.embeddings_by_alg[alg_id] = store.load_embeddings()
            print(f"  Algorithm {alg_id}: {self.embeddings_by_alg[alg_id].shape}")

        # Load raw tool counts and convert to dense matrix
        print("Loading raw tool counts...")
        self._load_raw_tool_counts_as_matrix()

    def _load_raw_tool_counts_as_matrix(self):
        """
        Load raw tool counts and convert sparse dictionary to dense numpy matrix.
        """
        raw_counts = CDistanceAnalysis_Baseline_Raw.load_raw_tool_counts_from_file()

        # Find all unique tool IDs across all users
        all_tool_ids = set()
        for user_id in self.all_user_ids:
            user_counts = raw_counts.get(user_id, {})
            all_tool_ids.update(user_counts.keys())

        # Create a sorted list of tool IDs for consistent ordering
        tool_id_list = sorted(list(all_tool_ids))
        tool_id_to_index = {tool_id: idx for idx, tool_id in enumerate(tool_id_list)}

        num_users = len(self.all_user_ids)
        num_tools = len(tool_id_list)

        print(f"  Raw tool counts: {num_users} users x {num_tools} tools")

        # Create dense matrix
        raw_matrix = np.zeros((num_users, num_tools), dtype=np.float32)

        for user_idx, user_id in enumerate(self.all_user_ids):
            user_counts = raw_counts.get(user_id, {})
            for tool_id, count in user_counts.items():
                tool_idx = tool_id_to_index[tool_id]
                raw_matrix[user_idx, tool_idx] = count

        self.embeddings_by_alg[self.RAW_ALG_ID] = raw_matrix

    def get_embedding_matrix(self, alg_id):
        """Get embedding matrix as numpy array for a given algorithm."""
        mat = self.embeddings_by_alg[alg_id]
        if isinstance(mat, torch.Tensor):
            return mat.numpy()
        return mat

    # ==================== WCSS ANALYSIS ====================
    def compute_wcss(self, alg_id, max_clusters=MAX_NUMBER_OF_CLUSTERS):
        """
        Compute Within-Cluster Sum of Squares (WCSS) for different cluster counts.
        Used to determine optimal number of clusters via elbow method.
        """
        mat = self.get_embedding_matrix(alg_id)
        no_of_samples = mat.shape[0]

        wcss = []
        cluster_range = range(2, min(max_clusters + 1, no_of_samples))

        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            kmeans.fit(mat)
            wcss.append(kmeans.inertia_)

        return list(cluster_range), wcss

    def print_wcss_for_all_algorithms(self, max_clusters=MAX_NUMBER_OF_CLUSTERS):
        """Print WCSS values for all algorithms and baselines."""
        print("\n" + "=" * 70)
        print("WCSS (Within-Cluster Sum of Squares) FOR OPTIMAL CLUSTER SELECTION")
        print("=" * 70)

        all_alg_ids = self.ALGORITHM_IDS + [self.RAW_ALG_ID]
        alg_names = {2: "Algorithm 2", 3: "Algorithm 3", 11: "PCA Baseline", 21: "Raw Tool Counts"}

        for alg_id in all_alg_ids:
            cluster_range, wcss = self.compute_wcss(alg_id, max_clusters)
            print(f"\n*** {alg_names[alg_id]} (ID: {alg_id}) ***")
            print(f"{'Clusters':<10} {'WCSS':<20}")
            print("-" * 30)
            for k, w in zip(cluster_range, wcss):
                print(f"{k:<10} {w:<20.4f}")

        print("=" * 70)

    # ==================== K-MEANS CLUSTERING ====================
    def compute_kmeans_clustering(self, alg_id, num_clusters=DEFAULT_NUMBER_OF_CLUSTERS):
        """
        Compute K-Means clustering for a given algorithm.
        Returns cluster labels and centroids.
        """
        mat = self.get_embedding_matrix(alg_id)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        kmeans.fit(mat)

        cluster_labels = kmeans.labels_
        cluster_centroids = kmeans.cluster_centers_

        return cluster_labels, cluster_centroids

    # ==================== SILHOUETTE SCORE ====================
    def compute_silhouette_score(self, alg_id, num_clusters=DEFAULT_NUMBER_OF_CLUSTERS):
        """
        Compute silhouette score for K-Means clustering.
        Silhouette score ranges from -1 to 1:
        - 1: Clusters are well separated
        - 0: Clusters are overlapping
        - -1: Samples are assigned to wrong clusters
        """
        mat = self.get_embedding_matrix(alg_id)

        # Need at least 2 clusters and more samples than clusters
        if num_clusters < 2 or mat.shape[0] <= num_clusters:
            return None

        cluster_labels, _ = self.compute_kmeans_clustering(alg_id, num_clusters)

        # Check if we have at least 2 unique labels
        unique_labels = len(set(cluster_labels))
        if unique_labels < 2:
            return None

        score = silhouette_score(mat, cluster_labels)
        return score

    def print_silhouette_scores_for_all_algorithms(self, num_clusters=DEFAULT_NUMBER_OF_CLUSTERS):
        """Print silhouette scores for all algorithms and baselines."""
        print("\n" + "=" * 70)
        print(f"SILHOUETTE SCORES (K-Means with {num_clusters} clusters)")
        print("=" * 70)
        print("Score interpretation: 1 = well separated, 0 = overlapping, -1 = wrong assignment")
        print("-" * 70)

        all_alg_ids = self.ALGORITHM_IDS + [self.RAW_ALG_ID]
        alg_names = {2: "Algorithm 2", 3: "Algorithm 3", 11: "PCA Baseline", 21: "Raw Tool Counts"}

        results = []
        for alg_id in all_alg_ids:
            score = self.compute_silhouette_score(alg_id, num_clusters)
            results.append((alg_id, alg_names[alg_id], score))

        print(f"\n{'Algorithm':<25} {'ID':<8} {'Silhouette Score':<20}")
        print("-" * 55)

        for alg_id, name, score in results:
            if score is not None:
                print(f"{name:<25} {alg_id:<8} {score:<20.6f}")
            else:
                print(f"{name:<25} {alg_id:<8} {'N/A':<20}")

        print("=" * 70)

    def print_silhouette_scores_for_range(self, min_clusters=2, max_clusters=MAX_NUMBER_OF_CLUSTERS):
        """Print silhouette scores for a range of cluster counts."""
        print("\n" + "=" * 70)
        print(f"SILHOUETTE SCORES FOR CLUSTER RANGE ({min_clusters} to {max_clusters})")
        print("=" * 70)

        all_alg_ids = self.ALGORITHM_IDS + [self.RAW_ALG_ID]
        alg_names = {2: "Alg 2", 3: "Alg 3", 11: "PCA", 21: "Raw"}

        # Print header
        header = f"{'Clusters':<10}"
        for alg_id in all_alg_ids:
            header += f"{alg_names[alg_id]:<15}"
        print(header)
        print("-" * (10 + 15 * len(all_alg_ids)))

        # Print scores for each cluster count
        for num_clusters in range(min_clusters, max_clusters + 1):
            row = f"{num_clusters:<10}"
            for alg_id in all_alg_ids:
                score = self.compute_silhouette_score(alg_id, num_clusters)
                if score is not None:
                    row += f"{score:<15.6f}"
                else:
                    row += f"{'N/A':<15}"
            print(row)

        print("=" * 70)

    # ==================== MAIN ANALYSIS ====================
    def print_full_analysis(self, num_clusters=DEFAULT_NUMBER_OF_CLUSTERS):
        """Run full clustering analysis."""
        print("\n" + "=" * 70)
        print("CLUSTERING ANALYSIS: ALGORITHMS VS BASELINES")
        print("=" * 70)

        # WCSS for optimal cluster selection
        self.print_wcss_for_all_algorithms()

        # Silhouette scores for specified number of clusters
        self.print_silhouette_scores_for_all_algorithms(num_clusters)

        # Silhouette scores for range of clusters
        self.print_silhouette_scores_for_range()


if __name__ == "__main__":
    print("Running Clustering Analysis for Algorithms and Baselines...")
    analysis = CClusteringAnalysis_Baseline()
    analysis.print_full_analysis(num_clusters=3)
