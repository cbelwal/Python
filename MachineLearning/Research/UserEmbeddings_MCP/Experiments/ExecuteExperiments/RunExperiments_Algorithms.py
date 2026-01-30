import os,sys
import datetime
# ----------------------------------------------
# Explicit declaration to ensure the root folder path is in sys.path 
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__))))
sys.path.append(topRootPath)
#----------------------------------------------

from Algorithms.Alg_1_DataPreparation import Algorithm_1_DataPreparation
from Algorithms.Alg_2_GenerateUserEmbeddings import  Algorithm_2_GenerateUserEmbeddings
from Algorithms.Alg_3_GenerateUserEmbeddings import Algorithm_3_GenerateUserEmbeddings
from Algorithms.Alg_Baseline_PCA_GenerateUserEmbeddings import  Alg_Baseline_PCA_GenerateUserEmbeddings

from Algorithms.Helpers.CDataMain import CDataMain
from Experiments.CConfig import CConfig
from Experiments.ExecuteExperiments.Helpers.CResultsStore import CResultsStore
from Experiments.ExecuteExperiments.Helpers.CDistanceAnalysis import CDistanceAnalysis


def run_algorithms_on_synthetic_data(All_C_hat_u:dict,algID:int):     
    # Step 2: Create Test Data Main Instance
    testData = CDataMain(All_C_hat_u)
    embeddingDimensions = CConfig.EMBEDDING_DIMENSIONS
    if(algID == 2):
        (MAT_E, loss_for_each_user) = Algorithm_2_GenerateUserEmbeddings(
                embeddingDimensions=embeddingDimensions,
                testData=testData)
    elif(algID == 3):
        (MAT_E, loss_for_each_user) = Algorithm_3_GenerateUserEmbeddings(
                embeddingDimensions=embeddingDimensions,
                testData=testData)
    print(f"Generated Embeddings with Algorithm {algID}") 
    return (MAT_E, loss_for_each_user)

def run_pca_on_synthetic_data(All_C_hat_u:dict,algID:int):     
    testData = CDataMain(All_C_hat_u)
    embeddingDimensions = CConfig.EMBEDDING_DIMENSIONS
   
    (MAT_E, loss_for_each_user) = Alg_Baseline_PCA_GenerateUserEmbeddings(
                embeddingDimensions=embeddingDimensions,
                testData=testData)
    print(f"Generated Embeddings with PCA") 
    return (MAT_E, loss_for_each_user)


def store_results_in_file(MAT_E, loss_for_each_user, algID:int):
    store = CResultsStore(algID=algID)
    # Store embeddings in file
    store.store_embeddings(MAT_E)
    print(f"User embeddings stored in file")
    # Store training loss in file
    store.store_training_loss(loss_for_each_user)
    print(f"Training loss stored in file")


# Run the experiments and store embeddings and losses in file
if __name__== "__main__":
    print(f"Starting experiments on synthetic data at time {datetime.datetime.now()}...")
    print("Running A/g #1 on synthetic data...")
    # Step 1: Data Preparation
    All_C_hat_u = Algorithm_1_DataPreparation()

    # Run algorithm 2
    algorithmsIds = [2,3]
    for algID in algorithmsIds:
        print(f"Computing embeddings for Algorithm {algID}...")
        (MAT_E, loss_for_each_user) = \
            run_algorithms_on_synthetic_data(All_C_hat_u,algID)
        print(f"Storing embeddings for Algorithm {algID}...")
        store_results_in_file(MAT_E, loss_for_each_user, algID=algID)
    print(f"Experiments completed at time {datetime.datetime.now()}.")

       
 