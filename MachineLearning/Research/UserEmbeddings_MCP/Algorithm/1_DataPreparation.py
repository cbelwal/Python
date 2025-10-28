import os,sys

# ----------------------------------------------
# Ensure the root folder path is in sys.path 
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__))))
sys.path.append(topRootPath)
#----------------------------------------------

from Algorithm.CDataPreparationHelper import CDataPreparationHelper


dataPrepHelper = CDataPreparationHelper()

def Algorithm_1_DataPreparation():
    All_C_hat_u = {} # Dictionary of dictionaries to hold C_hat_u for all users
    allUsersIds = dataPrepHelper.getAllUserIds()
    for userId in allUsersIds:
        C_u = {}
        C_hat_u = {}
        T_u = []
        sessionIdsForUser = dataPrepHelper.getUserSessions(userId)
        for sessionId in sessionIdsForUser:
            toolIds = getToolIdsFromSessionInteractions(sessionId)
            for toolId in toolIds:
                if toolId in C_u:
                    C_u[toolId] += 1
                else:
                    C_u[toolId] = 1
                if toolId not in T_u:
                    T_u.append(toolId)
        # Normalize tool calls by number of sessions
        Sum_C_u = 0    
        for toolId in T_u:
            C_u[toolId] = C_u[toolId] / len(sessionIdsForUser)
            Sum_C_u += C_u[toolId]
        
        for toolId in T_u:
            C_hat_u[toolId] = C_u[toolId] / Sum_C_u
        All_C_hat_u[userId] = C_hat_u
    return All_C_hat_u
  