import os,sys
import math
import pickle
# ----------------------------------------------
# Explicit declaration to ensure the root folder path is in sys.path
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(topRootPath)
#----------------------------------------------
from Experiments.Database.CDatabaseManager import CDatabaseManager
from Algorithms.Alg_Data_Raw import Algorithm_Data_Raw
from Experiments.CConfig import CConfig

class CDistanceAnalysis_Baseline_Raw:
    RAW_ALG_ID = 21  # Algorithm ID for raw tool counts

    def __init__(self):
        self.dbManager = CDatabaseManager()
        self.canary_users = self.dbManager.get_canary_users()
        self.all_user_ids = self.dbManager.get_all_user_ids()
        # Load raw tool counts (lazy load)
        self._raw_tool_counts = None

    # ==================== FILE STORAGE FUNCTIONS ====================
    @staticmethod
    def get_raw_tool_counts_file_path() -> str:
        """Get the file path for storing raw tool counts."""
        folderPath = os.path.dirname(
                     os.path.dirname(
                     os.path.dirname(
                     os.path.abspath(__file__))))
        resultsFolder = os.path.join(folderPath, "Data", "ExperimentResults")
        # Use the training loss file name pattern with alg_21
        baseFileName = CConfig.BASE_EMBEDDINGS_FILE_NAME
        fileNameComponents = baseFileName.split(".")
        fileName = f"{fileNameComponents[0]}_alg_{CDistanceAnalysis_Baseline_Raw.RAW_ALG_ID}.{fileNameComponents[1]}"
        filePath = os.path.join(resultsFolder, fileName)
        return filePath

    @staticmethod
    def store_raw_tool_counts():
        """
        Read raw tool counts from DB and store to file.
        This should be run manually once to create the file.
        """
        print("Reading raw tool counts from database...")
        raw_tool_counts = Algorithm_Data_Raw()

        filePath = CDistanceAnalysis_Baseline_Raw.get_raw_tool_counts_file_path()
        print(f"Storing raw tool counts to file: {filePath}")

        with open(filePath, 'wb') as f:
            pickle.dump(raw_tool_counts, f)

        print(f"Raw tool counts stored successfully. Total users: {len(raw_tool_counts)}")

    @staticmethod
    def load_raw_tool_counts_from_file():
        """Load raw tool counts from the stored file."""
        filePath = CDistanceAnalysis_Baseline_Raw.get_raw_tool_counts_file_path()
        print(f"Loading raw tool counts from file: {filePath}")

        with open(filePath, 'rb') as f:
            raw_tool_counts = pickle.load(f)

        print (f"Raw tool counts loaded successfully. Total users: {len(raw_tool_counts)}")
        return raw_tool_counts

    def get_raw_tool_counts(self):
        """Lazy load raw tool counts from stored file."""
        if self._raw_tool_counts is None:
            self._raw_tool_counts = self.load_raw_tool_counts_from_file()
        return self._raw_tool_counts

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

    # ==================== RAW TOOL COUNT DISTANCE FUNCTIONS ====================
    @staticmethod
    def cosine_distance_sparse(vec1_dict, vec2_dict):
        """
        Compute cosine distance between two sparse vectors (dictionaries).
        Returns 1 - cosine_similarity.
        """
        # Get all unique keys
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

    def compute_pairwise_distances_raw(self, user_ids_1, user_ids_2, raw_counts):
        """
        Compute all pairwise cosine and euclidean distances between two groups of users
        using raw tool counts.
        """
        cosine_distances = []
        euclidean_distances = []

        for user_id_1 in user_ids_1:
            vec1 = raw_counts.get(user_id_1, {})
            for user_id_2 in user_ids_2:
                if user_id_1 == user_id_2:
                    continue
                vec2 = raw_counts.get(user_id_2, {})

                cosine_dist = self.cosine_distance_sparse(vec1, vec2)
                euclidean_dist = self.euclidean_distance_sparse(vec1, vec2)

                cosine_distances.append(cosine_dist)
                euclidean_distances.append(euclidean_dist)

        return cosine_distances, euclidean_distances

    # ==================== CANARY GROUP ANALYSIS ====================
    def compute_average_distances_within_canary_group_raw(self, canary_id):
        """
        Compute average distances within a single canary group using raw tool counts.
        """
        canary_users = self.get_canary_user_ids(canary_id)

        if len(canary_users) < 2:
            print(f"Error: Not enough canary users in category {canary_id}.")
            return

        raw_counts = self.get_raw_tool_counts()

        print(f"\n*** RAW TOOL COUNT Distances within Canary {canary_id} group ({len(canary_users)} users) ***")

        cosine_distances, euclidean_distances = self.compute_pairwise_distances_raw(
            canary_users, canary_users, raw_counts
        )

        if len(cosine_distances) > 0:
            avg_cosine = sum(cosine_distances) / len(cosine_distances)
            avg_euclidean = sum(euclidean_distances) / len(euclidean_distances)

            print(f"Raw Counts : Avg Cosine Distance = {self.format_distance(avg_cosine)}, "
                  f"Avg Euclidean Distance = {self.format_distance(avg_euclidean)}")

    # ==================== BETWEEN CANARY GROUPS ANALYSIS ====================
    def compute_average_distances_between_canary_groups_raw(self):
        """
        Compute average cosine and euclidean distances between Canary 1 and Canary 2 users
        using raw tool counts.
        """
        canary_1_users = self.get_canary_user_ids(1)
        canary_2_users = self.get_canary_user_ids(2)

        if len(canary_1_users) == 0 or len(canary_2_users) == 0:
            print("Error: Not enough canary users found in one or both categories.")
            return

        raw_counts = self.get_raw_tool_counts()

        print("=" * 70)
        print("RAW TOOL COUNT: AVERAGE DISTANCES BETWEEN CANARY 1 AND CANARY 2 USERS")
        print("=" * 70)
        print(f"Canary 1 users: {len(canary_1_users)}, Canary 2 users: {len(canary_2_users)}")
        print(f"Total pairs: {len(canary_1_users) * len(canary_2_users)}")
        print("-" * 70)

        cosine_distances, euclidean_distances = self.compute_pairwise_distances_raw(
            canary_1_users, canary_2_users, raw_counts
        )

        avg_cosine = sum(cosine_distances) / len(cosine_distances)
        avg_euclidean = sum(euclidean_distances) / len(euclidean_distances)

        print(f"Raw Counts : Avg Cosine Distance = {self.format_distance(avg_cosine)}, "
              f"Avg Euclidean Distance = {self.format_distance(avg_euclidean)}")

        print("=" * 70)

    # ==================== ALL USERS ANALYSIS ====================
    def compute_average_distances_all_users_raw(self):
        """
        Compute average cosine and euclidean distances between all users
        using raw tool counts.
        """
        all_users = self.all_user_ids

        if len(all_users) < 2:
            print("Error: Not enough users found in database.")
            return

        raw_counts = self.get_raw_tool_counts()

        print("\n" + "=" * 70)
        print("RAW TOOL COUNT: AVERAGE DISTANCES BETWEEN ALL USERS")
        print("=" * 70)
        print(f"Total users: {len(all_users)}")
        print(f"Total pairs: {len(all_users) * (len(all_users) - 1) // 2}")
        print("-" * 70)

        cosine_distances = []
        euclidean_distances = []

        # Compute pairwise distances for all unique pairs
        for i in range(len(all_users)):
            user_id_1 = all_users[i]
            vec1 = raw_counts.get(user_id_1, {})

            for j in range(i + 1, len(all_users)):
                user_id_2 = all_users[j]
                vec2 = raw_counts.get(user_id_2, {})

                cosine_dist = self.cosine_distance_sparse(vec1, vec2)
                euclidean_dist = self.euclidean_distance_sparse(vec1, vec2)

                cosine_distances.append(cosine_dist)
                euclidean_distances.append(euclidean_dist)

        avg_cosine = sum(cosine_distances) / len(cosine_distances)
        avg_euclidean = sum(euclidean_distances) / len(euclidean_distances)

        print(f"Raw Counts : Avg Cosine Distance = {self.format_distance(avg_cosine)}, "
              f"Avg Euclidean Distance = {self.format_distance(avg_euclidean)}")

        print("=" * 70)

    # ==================== MAIN ANALYSIS FUNCTION ====================
    def print_all_raw_analysis(self):
        """
        Print comprehensive baseline analysis for raw tool counts including:
        - Distances within Canary 1 group
        - Distances within Canary 2 group
        - Distances between Canary 1 and Canary 2 groups
        - Distances between all users
        """
        print("\n" + "=" * 70)
        print("RAW TOOL COUNT BASELINE ANALYSIS")
        print("=" * 70 + "\n")

        # ---- CANARY 1 WITHIN GROUP ----
        self.compute_average_distances_within_canary_group_raw(1)

        # ---- CANARY 2 WITHIN GROUP ----
        self.compute_average_distances_within_canary_group_raw(2)

        # ---- ALL USERS ----
        #self.compute_average_distances_all_users_raw()


if __name__ == "__main__":
    print("Running Raw Tool Count Baseline Analysis...")
    analysis = CDistanceAnalysis_Baseline_Raw()
    #CDistanceAnalysis_Baseline_Raw.store_raw_tool_counts()
    analysis.print_all_raw_analysis()
