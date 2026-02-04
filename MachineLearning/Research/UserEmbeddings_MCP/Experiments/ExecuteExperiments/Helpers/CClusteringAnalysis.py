"""
Group into clusters based on embedding distances
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
from sklearn.cluster import KMeans
from kneed import KneeLocator
import torch
import numpy as np

MAX_NUMBER_OF_CLUSTERS = 8
GIVEN_NUMBER_OF_CLUSTERS = 1

class CClusteringAnalysis:
    def __init__(self, MAT_E):
        self.MAT_E = MAT_E
        self.dbManager = CDatabaseManager()
        self.canary_users = self.dbManager.get_canary_users()
        self.all_user_ids = self.dbManager.get_all_user_ids()

    # Generate number of clusters using elbow method
    # Goal is to minimize WCSS (within cluster sum of squares)
    def determine_optimal_number_of_clusters_elbow(self, max_clusters:int=MAX_NUMBER_OF_CLUSTERS):
        # Within-cluster sum of squares (WCSS) / Inertia
        wcss = []  # Within-cluster sum of squares
        no_of_samples = self.MAT_E.shape[0]
        for n_clusters in range(1, max_clusters + 1):
            if n_clusters > no_of_samples:
                break
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
            kmeans.fit(self.MAT_E.numpy())
            wcss.append(kmeans.inertia_)
        
        optimal_clusters = self._find_elbow_point(wcss, max_clusters)
        print(f"Optimal number of clusters (elbow method): {optimal_clusters}")

        return wcss

    def _find_elbow_point(self, wcss: list, max_clusters: int) -> int:
        """
        Determine the optimal number of clusters by finding the elbow point in WCSS.
        Uses the kneed library's KneeLocator to detect the elbow.
        """
        k_range = range(1, len(wcss) + 1)
        kneedle = KneeLocator(
            x=list(k_range),
            y=wcss,
            curve='convex',
            direction='decreasing'
        )

        # Return the elbow point, default to 2 if no elbow is found
        return kneedle.elbow if kneedle.elbow is not None else 2

    # Default distance metric is Euclidean
    def generate_kmeans_clustering(self,num_clusters:int=GIVEN_NUMBER_OF_CLUSTERS):
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10) # n_init for robust centroid initialization
        
        print("Fitting KMeans clustering...")
        kmeans.fit(self.MAT_E) #.numpy())
        cluster_labels = kmeans.labels_

        # Get the coordinates of the cluster centroids
        cluster_centroids = kmeans.cluster_centers_
        #print("Cluster Centroids:", cluster_centroids)

        return cluster_labels, cluster_centroids
    
if __name__ == "__main__":
    MAT_E = torch.zeros(3, 3) # 2 users x 3 tools
    # ------ Manually Assign Values
    MAT_E[0][0] = 1 # user 0, tool 0
    MAT_E[0][1] = 2 # user 0, tool 1
    MAT_E[0][2] = 2 # user 0, tool 2
    MAT_E[1][0] = 3 # user 1, tool 0
    MAT_E[1][1] = 1 # user 1, tool 1
    MAT_E[1][2] = 0 # user 1, tool 2
    MAT_E[2][0] = 4 # user 1, tool 0
    MAT_E[2][1] = 3 # user 1, tool 1
    MAT_E[2][2] = 1 # user 1, tool 2
    
    analysis = CClusteringAnalysis(MAT_E)
    print("WCSS:",analysis.determine_optimal_number_of_clusters_elbow())
    
    #(cluster_labels, cluster_centroids) = analysis.generate_kmeans_clustering()   
    #print(cluster_centroids)    