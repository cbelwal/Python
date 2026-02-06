# Computes Z-Normalized Cosine and Euclidean distances for embeddings from:
# Algorithms 2, 3, PCA Baseline (11), and Raw Tool Counts (21)
# All 4 algorithms are compared using a single pooled reference for fair comparison.
import os,sys
import math
# ----------------------------------------------
# Explicit declaration to ensure the root folder path is in sys.path
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(topRootPath)
#----------------------------------------------
from Experiments.ExecuteExperiments.Helpers.CDistanceFunctions import CDistanceFunctions
from Experiments.ExecuteExperiments.Helpers.CResultsStore import CResultsStore
from Experiments.ExecuteExperiments.Helpers.CZScoreWithPooledRef import CZScoreWithPooledRef
from Experiments.Database.CDatabaseManager import CDatabaseManager
from Algorithms.Alg_Data_Raw import Algorithm_Data_Raw

class CDistanceAnalysis_Baseline:
    # Algorithms with tensor embeddings
    TENSOR_ALGORITHM_IDS = [2, 3, 11]
    # Raw tool counts algorithm (uses sparse dictionary format)
    RAW_ALG_ID = 21
    # All algorithms for display
    ALL_ALGORITHM_IDS = [2, 3, 11, 21]

    def __init__(self):
        self.dbManager = CDatabaseManager()
        self.canary_users = self.dbManager.get_canary_users()
        self.all_user_ids = self.dbManager.get_all_user_ids()

        # Load tensor embeddings for algorithms 2, 3, 11
        self.embeddings_by_alg = {}
        for alg_id in self.TENSOR_ALGORITHM_IDS:
            store = CResultsStore(algID=alg_id)
            self.embeddings_by_alg[alg_id] = store.load_embeddings()

        # Load raw tool counts shape for display (lazy load actual data)
        store = CResultsStore(self.RAW_ALG_ID)
        self._raw_tensor_shape = tuple(store.load_embeddings().shape)
        self._raw_tool_counts = None

    def get_canary_user_ids(self, canary_id):
        """Get all user IDs for a given canary category."""
        return self.canary_users.get(canary_id, [])

    def get_raw_tool_counts(self):
        """Lazy load raw tool counts as sparse dictionary from database."""
        if self._raw_tool_counts is None:
            self._raw_tool_counts = Algorithm_Data_Raw()
        return self._raw_tool_counts

    @staticmethod
    def format_distance(value):
        """Format distance value to handle very small numbers."""
        if value == 0:
            return "0.000000"
        elif abs(value) < 0.000001:
            return f"{value:.6e}"
        else:
            return f"{value:.6f}"

    # ==================== RAW TOOL COUNT DISTANCE FUNCTIONS ====================
    @staticmethod
    def cosine_distance_sparse(vec1_dict, vec2_dict):
        """
        Compute cosine distance between two sparse vectors (dictionaries).
        Returns 1 - cosine_similarity.
        """
        all_keys = set(vec1_dict.keys()) | set(vec2_dict.keys())

        dot_product = 0.0
        magnitude_1 = 0.0
        magnitude_2 = 0.0

        for key in all_keys:
            val1 = vec1_dict.get(key, 0)
            val2 = vec2_dict.get(key, 0)
            dot_product += val1 * val2
            magnitude_1 += val1 * val1
            magnitude_2 += val2 * val2

        magnitude_1 = math.sqrt(magnitude_1)
        magnitude_2 = math.sqrt(magnitude_2)

        if magnitude_1 == 0 or magnitude_2 == 0:
            return 1.0  # Maximum distance if one vector is zero

        cosine_similarity = dot_product / (magnitude_1 * magnitude_2)
        return 1.0 - cosine_similarity

    @staticmethod
    def euclidean_distance_sparse(vec1_dict, vec2_dict):
        """
        Compute Euclidean distance between two sparse vectors (dictionaries).
        """
        all_keys = set(vec1_dict.keys()) | set(vec2_dict.keys())

        sum_squared_diff = 0.0
        for key in all_keys:
            val1 = vec1_dict.get(key, 0)
            val2 = vec2_dict.get(key, 0)
            sum_squared_diff += (val1 - val2) ** 2

        return math.sqrt(sum_squared_diff)

    # ==================== EMBEDDING DISTANCE FUNCTIONS ====================
    def compute_pairwise_distances_for_user_group(self, user_ids, MAT_E):
        """
        Compute all pairwise cosine and euclidean distances within a group of users
        using tensor embeddings.
        Returns lists of cosine_distances and euclidean_distances.
        """
        cosine_distances = []
        euclidean_distances = []

        for i in range(len(user_ids)):
            user_id_1 = user_ids[i]
            # CAUTION: MAT_E is 0-indexed, user IDs are 1-indexed in DB
            embedding_1 = MAT_E[user_id_1 - 1]

            for j in range(i + 1, len(user_ids)):
                user_id_2 = user_ids[j]
                embedding_2 = MAT_E[user_id_2 - 1]

                cosine_dist = CDistanceFunctions.cosine_distance_tensors(embedding_1, embedding_2)
                euclidean_dist = CDistanceFunctions.euclidean_distance_tensors(embedding_1, embedding_2)

                cosine_distances.append(cosine_dist.item())
                euclidean_distances.append(euclidean_dist.item())

        return cosine_distances, euclidean_distances

    def compute_pairwise_distances_for_user_group_raw(self, user_ids):
        """
        Compute all pairwise cosine and euclidean distances within a group of users
        using raw tool counts (sparse dictionary format).
        Returns lists of cosine_distances and euclidean_distances.
        """
        raw_counts = self.get_raw_tool_counts()
        cosine_distances = []
        euclidean_distances = []

        for i in range(len(user_ids)):
            user_id_1 = user_ids[i]
            vec1 = raw_counts.get(user_id_1, {})

            for j in range(i + 1, len(user_ids)):
                user_id_2 = user_ids[j]
                vec2 = raw_counts.get(user_id_2, {})

                cosine_dist = self.cosine_distance_sparse(vec1, vec2)
                euclidean_dist = self.euclidean_distance_sparse(vec1, vec2)

                cosine_distances.append(cosine_dist)
                euclidean_distances.append(euclidean_dist)

        return cosine_distances, euclidean_distances

    def _collect_distances_for_all_algorithms(self, user_ids, description):
        """
        Helper function to collect pairwise distances for ALL algorithms (2, 3, 11, 21).
        Returns cosine_datasets and euclidean_datasets dictionaries.
        """
        print(f"\n  Computing pairwise distances for {description}...")
        print(f"  Users: {len(user_ids)}, Pairs per algorithm: {len(user_ids) * (len(user_ids) - 1) // 2}")

        cosine_datasets = {}
        euclidean_datasets = {}

        # Collect distances for tensor-based algorithms (2, 3, 11)
        for alg_id in self.TENSOR_ALGORITHM_IDS:
            MAT_E = self.embeddings_by_alg[alg_id]
            shape = tuple(MAT_E.shape)

            cosine_distances, euclidean_distances = self.compute_pairwise_distances_for_user_group(
                user_ids, MAT_E
            )

            key = f"Alg_{alg_id} {shape}"
            cosine_datasets[key] = cosine_distances
            euclidean_datasets[key] = euclidean_distances

        # Collect distances for raw tool counts (algorithm 21)
        cosine_distances_raw, euclidean_distances_raw = self.compute_pairwise_distances_for_user_group_raw(
            user_ids
        )
        key_raw = f"Alg_{self.RAW_ALG_ID} {self._raw_tensor_shape}"
        cosine_datasets[key_raw] = cosine_distances_raw
        euclidean_datasets[key_raw] = euclidean_distances_raw

        return cosine_datasets, euclidean_datasets

    # ==================== Z-NORMALIZED CANARY 1 ANALYSIS ====================
    def compute_z_normalized_distances_canary_1(self, plot_kde: bool = False,
                                                  save_plot_path: str = None):
        """
        Compute z-normalized statistics for pairwise distances within Canary 1 group.
        Uses pooled reference from ALL algorithms (2, 3, 11, 21) for fair comparison.

        Args:
            plot_kde: If True, plot KDE of z-values for each algorithm
            save_plot_path: Optional path prefix to save the KDE plot
        """
        canary_users = self.get_canary_user_ids(1)

        if len(canary_users) < 2:
            print(f"Error: Not enough canary users in category 1.")
            return

        print("\n" + "=" * 80)
        print("Z-NORMALIZED DISTANCES WITHIN CANARY 1 GROUP (Algorithms 2, 3, 11, 21)")
        print("=" * 80)

        # Collect distances for all algorithms
        cosine_datasets, euclidean_datasets = self._collect_distances_for_all_algorithms(
            canary_users, "Canary 1 group"
        )

        # Compute and print z-normalized statistics for cosine distances
        cosine_normalizer = CZScoreWithPooledRef()
        cosine_normalizer.print_z_normalized_stats(
            cosine_datasets,
            "Z-NORMALIZED COSINE DISTANCES - CANARY 1"
        )

        # Compute and print z-normalized statistics for euclidean distances
        euclidean_normalizer = CZScoreWithPooledRef()
        euclidean_normalizer.print_z_normalized_stats(
            euclidean_datasets,
            "Z-NORMALIZED EUCLIDEAN DISTANCES - CANARY 1"
        )

        # Optional: Plot KDE
        if plot_kde:
            print("\nGenerating KDE plots for Canary 1...")
            cosine_normalizer.plot_kde_all_datasets(
                cosine_datasets,
                "KDE of Z-Normalized Cosine Distances - Canary 1",
                save_path=f"{save_plot_path}_canary1_cosine.png" if save_plot_path else None
            )
            euclidean_normalizer.plot_kde_all_datasets(
                euclidean_datasets,
                "KDE of Z-Normalized Euclidean Distances - Canary 1",
                save_path=f"{save_plot_path}_canary1_euclidean.png" if save_plot_path else None
            )

    # ==================== Z-NORMALIZED CANARY 2 ANALYSIS ====================
    def compute_z_normalized_distances_canary_2(self, plot_kde: bool = False,
                                                  save_plot_path: str = None):
        """
        Compute z-normalized statistics for pairwise distances within Canary 2 group.
        Uses pooled reference from ALL algorithms (2, 3, 11, 21) for fair comparison.

        Args:
            plot_kde: If True, plot KDE of z-values for each algorithm
            save_plot_path: Optional path prefix to save the KDE plot
        """
        canary_users = self.get_canary_user_ids(2)

        if len(canary_users) < 2:
            print(f"Error: Not enough canary users in category 2.")
            return

        print("\n" + "=" * 80)
        print("Z-NORMALIZED DISTANCES WITHIN CANARY 2 GROUP (Algorithms 2, 3, 11, 21)")
        print("=" * 80)

        # Collect distances for all algorithms
        cosine_datasets, euclidean_datasets = self._collect_distances_for_all_algorithms(
            canary_users, "Canary 2 group"
        )

        # Compute and print z-normalized statistics for cosine distances
        cosine_normalizer = CZScoreWithPooledRef()
        cosine_normalizer.print_z_normalized_stats(
            cosine_datasets,
            "Z-NORMALIZED COSINE DISTANCES - CANARY 2"
        )

        # Compute and print z-normalized statistics for euclidean distances
        euclidean_normalizer = CZScoreWithPooledRef()
        euclidean_normalizer.print_z_normalized_stats(
            euclidean_datasets,
            "Z-NORMALIZED EUCLIDEAN DISTANCES - CANARY 2"
        )

        # Optional: Plot KDE
        if plot_kde:
            print("\nGenerating KDE plots for Canary 2...")
            cosine_normalizer.plot_kde_all_datasets(
                cosine_datasets,
                "KDE of Z-Normalized Cosine Distances - Canary 2",
                save_path=f"{save_plot_path}_canary2_cosine.png" if save_plot_path else None
            )
            euclidean_normalizer.plot_kde_all_datasets(
                euclidean_datasets,
                "KDE of Z-Normalized Euclidean Distances - Canary 2",
                save_path=f"{save_plot_path}_canary2_euclidean.png" if save_plot_path else None
            )

    # ==================== Z-NORMALIZED ALL USERS ANALYSIS ====================
    def compute_z_normalized_distances_all_users(self, plot_kde: bool = False,
                                                   save_plot_path: str = None):
        """
        Compute z-normalized statistics for pairwise distances across all users.
        Uses pooled reference from ALL algorithms (2, 3, 11, 21) for fair comparison.

        Args:
            plot_kde: If True, plot KDE of z-values for each algorithm
            save_plot_path: Optional path prefix to save the KDE plot
        """
        all_users = self.all_user_ids

        if len(all_users) < 2:
            print("Error: Not enough users found in database.")
            return

        print("\n" + "=" * 80)
        print("Z-NORMALIZED DISTANCES FOR ALL USERS (Algorithms 2, 3, 11, 21)")
        print("=" * 80)

        # Collect distances for all algorithms
        cosine_datasets, euclidean_datasets = self._collect_distances_for_all_algorithms(
            all_users, "all users"
        )

        # Compute and print z-normalized statistics for cosine distances
        cosine_normalizer = CZScoreWithPooledRef()
        cosine_normalizer.print_z_normalized_stats(
            cosine_datasets,
            "Z-NORMALIZED COSINE DISTANCES - ALL USERS"
        )

        # Compute and print z-normalized statistics for euclidean distances
        euclidean_normalizer = CZScoreWithPooledRef()
        euclidean_normalizer.print_z_normalized_stats(
            euclidean_datasets,
            "Z-NORMALIZED EUCLIDEAN DISTANCES - ALL USERS"
        )

        # Optional: Plot KDE
        if plot_kde:
            print("\nGenerating KDE plots for all users...")
            cosine_normalizer.plot_kde_all_datasets(
                cosine_datasets,
                "KDE of Z-Normalized Cosine Distances - All Users",
                save_path=f"{save_plot_path}_all_users_cosine.png" if save_plot_path else None
            )
            euclidean_normalizer.plot_kde_all_datasets(
                euclidean_datasets,
                "KDE of Z-Normalized Euclidean Distances - All Users",
                save_path=f"{save_plot_path}_all_users_euclidean.png" if save_plot_path else None
            )

    # ==================== MAIN ANALYSIS FUNCTION ====================
    def print_all_baseline_analysis(self, plot_kde: bool = False,
                                     save_plot_path: str = None,
                                     include_all_users: bool = False):
        """
        Print comprehensive z-normalized baseline analysis including:
        - Z-normalized distances within Canary 1 group
        - Z-normalized distances within Canary 2 group
        - Z-normalized distances for all users (optional, slow)

        All 4 algorithms (2, 3, 11, 21) are compared using a single pooled reference.

        Args:
            plot_kde: If True, plot KDE of z-values for each algorithm
            save_plot_path: Optional path prefix to save KDE plots
            include_all_users: If True, compute z-normalized distances for all users
                              (can be very slow for large datasets)
        """
        print("\n" + "=" * 80)
        print("   Z-NORMALIZED EMBEDDING BASELINE ANALYSIS (Algorithms 2, 3, 11, 21)")
        print("=" * 80 + "\n")

        # ---- CANARY 1 Z-NORMALIZED ----
        self.compute_z_normalized_distances_canary_1(
            plot_kde=plot_kde,
            save_plot_path=save_plot_path
        )

        # ---- CANARY 2 Z-NORMALIZED ----
        self.compute_z_normalized_distances_canary_2(
            plot_kde=plot_kde,
            save_plot_path=save_plot_path
        )

        # ---- ALL USERS Z-NORMALIZED (optional, slow) ----
        if include_all_users:
            self.compute_z_normalized_distances_all_users(
                plot_kde=plot_kde,
                save_plot_path=save_plot_path
            )
        else:
            print("\n  [Skipping all users analysis - set include_all_users=True to enable]")


if __name__ == "__main__":
    print("Running Z-Normalized Embedding Baseline Analysis (All 4 Algorithms)...")
    analysis = CDistanceAnalysis_Baseline()
    # Set plot_kde=True to generate KDE plots
    # Set include_all_users=True to compute for all users (slow)
    analysis.print_all_baseline_analysis(plot_kde=True, include_all_users=False)
