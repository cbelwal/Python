import os,sys
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
from Experiments.Database.CDatabaseManager import CDatabaseManager

class CDistanceAnalysis_Baseline_PCA:
    ALGORITHM_IDS = [2, 3, 11]  # Algorithms to analyze

    def __init__(self):
        self.dbManager = CDatabaseManager()
        self.canary_users = self.dbManager.get_canary_users()
        self.all_user_ids = self.dbManager.get_all_user_ids()
        # Load embeddings for each algorithm
        self.embeddings_by_alg = {}
        for alg_id in self.ALGORITHM_IDS:
            store = CResultsStore(algID=alg_id)
            self.embeddings_by_alg[alg_id] = store.load_embeddings()

    def get_canary_user_ids(self, canary_id):
        """Get all user IDs for a given canary category."""
        return self.canary_users.get(canary_id, [])

    @staticmethod
    def format_distance(value):
        """Format distance value to handle very small numbers."""
        if value == 0:
            return "0.000000"
        elif abs(value) < 0.000001:
            return f"{value:.6e}"
        else:
            return f"{value:.6f}"

    # ==================== EMBEDDING DISTANCE FUNCTIONS ====================
    def compute_pairwise_distances(self, user_ids_1, user_ids_2, MAT_E):
        """
        Compute all pairwise cosine and euclidean distances between two groups of users.
        Returns lists of (cosine_distance, euclidean_distance) for each pair.
        """
        cosine_distances = []
        euclidean_distances = []

        for user_id_1 in user_ids_1:
            # CAUTION: MAT_E is 0-indexed, user IDs are 1-indexed in DB
            embedding_1 = MAT_E[user_id_1 - 1]
            for user_id_2 in user_ids_2:
                if user_id_1 == user_id_2:
                    continue  # Skip self-comparisons
                embedding_2 = MAT_E[user_id_2 - 1]

                cosine_dist = CDistanceFunctions.cosine_distance_tensors(embedding_1, embedding_2)
                euclidean_dist = CDistanceFunctions.euclidean_distance_tensors(embedding_1, embedding_2)

                cosine_distances.append(cosine_dist.item())
                euclidean_distances.append(euclidean_dist.item())

        return cosine_distances, euclidean_distances

    # ==================== CANARY GROUP ANALYSIS ====================
    def compute_average_distances_within_canary_group(self, canary_id):
        """
        Compute average distances within a single canary group.
        """
        canary_users = self.get_canary_user_ids(canary_id)

        if len(canary_users) < 2:
            print(f"Error: Not enough canary users in category {canary_id}.")
            return

        print(f"\n*** Distances within Canary {canary_id} group ({len(canary_users)} users) ***")

        for alg_id in self.ALGORITHM_IDS:
            MAT_E = self.embeddings_by_alg[alg_id]

            # Compute pairwise distances within the same group
            cosine_distances, euclidean_distances = self.compute_pairwise_distances(
                canary_users, canary_users, MAT_E
            )

            if len(cosine_distances) > 0:
                avg_cosine = sum(cosine_distances) / len(cosine_distances)
                avg_euclidean = sum(euclidean_distances) / len(euclidean_distances)

                print(f"Algorithm {alg_id:2d}: Avg Cosine Distance = {self.format_distance(avg_cosine)}, "
                      f"Avg Euclidean Distance = {self.format_distance(avg_euclidean)}")


    # ==================== ALL USERS ANALYSIS ====================
    def compute_average_distances_all_users(self):
        """
        Compute average cosine and euclidean distances between all users
        for all algorithms (2, 3, and 11).
        """
        all_users = self.all_user_ids

        if len(all_users) < 2:
            print("Error: Not enough users found in database.")
            return

        print("\n" + "=" * 70)
        print("AVERAGE DISTANCES BETWEEN ALL USERS")
        print("=" * 70)
        print(f"Total users: {len(all_users)}")
        print(f"Total pairs: {len(all_users) * (len(all_users) - 1) // 2}")
        print("-" * 70)

        for alg_id in self.ALGORITHM_IDS:
            MAT_E = self.embeddings_by_alg[alg_id]

            cosine_distances = []
            euclidean_distances = []

            # Compute pairwise distances for all unique pairs
            for i in range(len(all_users)):
                user_id_1 = all_users[i]
                embedding_1 = MAT_E[user_id_1 - 1]  # 0-indexed

                for j in range(i + 1, len(all_users)):
                    user_id_2 = all_users[j]
                    embedding_2 = MAT_E[user_id_2 - 1]  # 0-indexed

                    cosine_dist = CDistanceFunctions.cosine_distance_tensors(embedding_1, embedding_2)
                    euclidean_dist = CDistanceFunctions.euclidean_distance_tensors(embedding_1, embedding_2)

                    cosine_distances.append(cosine_dist.item())
                    euclidean_distances.append(euclidean_dist.item())

            avg_cosine = sum(cosine_distances) / len(cosine_distances)
            avg_euclidean = sum(euclidean_distances) / len(euclidean_distances)

            print(f"Algorithm {alg_id:2d}: Avg Cosine Distance = {self.format_distance(avg_cosine)}, "
                  f"Avg Euclidean Distance = {self.format_distance(avg_euclidean)}")

        print("=" * 70)

    # ==================== MAIN ANALYSIS FUNCTION ====================
    def print_all_baseline_analysis(self):
        """
        Print comprehensive baseline analysis including:
        - Distances within Canary 1 group
        - Distances within Canary 2 group
        - Distances between Canary 1 and Canary 2 groups
        - Distances between all users
        """
        print("\n" + "=" * 70)
        print("EMBEDDING BASELINE ANALYSIS (Algorithms 2, 3, 11)")
        print("=" * 70 + "\n")

        # ---- CANARY 1 WITHIN GROUP ----
        self.compute_average_distances_within_canary_group(1)

        # ---- CANARY 2 WITHIN GROUP ----
        self.compute_average_distances_within_canary_group(2)

        # ---- ALL USERS ----
        #self.compute_average_distances_all_users()


if __name__ == "__main__":
    print("Running Embedding Baseline Analysis...")
    analysis = CDistanceAnalysis_Baseline_PCA()
    analysis.print_all_baseline_analysis()
