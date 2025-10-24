'''
This class will create synthetic data for testing the user embeddings model.

and insert it into the database.
'''
import os,sys
import random
import numpy as np
from tqdm import tqdm

# Ensure the root folder path is in sys.path ---
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__))))
sys.path.append(topRootPath)
#----------------------------------------------
from UserEmbeddings_MCP.Experiments.CConfig import CConfig
from UserEmbeddings_MCP.Experiments.Database.CDatabaseManager import CDatabaseManager

class CGenerateSyntheticData:
    def __init__(self):
        self.dbManager = CDatabaseManager()
        self.dbManager.create_tables()
        random.seed(42)  # For reproducibility

    def __create_users__(self):
        print("Creating users...")
        for user_id in tqdm(range(1, CConfig.MAX_USERS+1)):
            insert_user_query = "INSERT INTO users (id) VALUES (?);"
            self.dbManager.execute_query(insert_user_query, (user_id,))
        print("All Users created.")

    def __create_mcp_servers_and_tools__(self):
        print("Creating MCP servers and tools...")
        for mcp_server_id in tqdm(range(1, CConfig.MAX_MCP_SERVERS+1)):
            no_of_tools = random.randint(CConfig.MIN_TOOLS_PER_MCP, CConfig.MAX_TOOLS_PER_MCP)
            insert_mcp_query = "INSERT INTO mcp_servers (id, no_of_tools) VALUES (?, ?);"
            self.dbManager.execute_query(insert_mcp_query, (mcp_server_id, no_of_tools))
            for tool_id in range(1, no_of_tools+1):
                insert_tool_query = "INSERT INTO mcp_tools (mcp_server_id, mcp_tool_id) VALUES (?, ?);"
                self.dbManager.execute_query(insert_tool_query,(mcp_server_id, tool_id))
        print("MCP servers and tools created.")

    # Before running this, ensure users and MCP Servers are created
    def __create_sessions_and_interactions_for_user__(self, user_id):
        num_sessions = max(1, int(np.random.normal(CConfig.SESSIONS_PER_USER_MEAN, CConfig.SESSIONS_PER_USER_STD)))
        for session_index in range(num_sessions):
            session_length = max(1, int(np.random.normal(CConfig.SESSIONS_LENGTH_MEAN, CConfig.SESSIONS_LENGTH_STD)))
            insert_session_query = "INSERT INTO sessions (user_id, session_depth) VALUES (?, ?);"
            session_id = self.dbManager.execute_query(insert_session_query, (user_id, session_length))
            #(session_id,) = self.dbManager.execute_read_query("SELECT last_insert_rowid();")[0] #.fetchone()[0]
            #session_id = self.dbManager.last_insert_rowid()

            # Now insert into session_details
            mcp_server_id = random.randint(1, CConfig.MAX_MCP_SERVERS)
            for seq_num in range(session_length):
                no_of_tools = self.__get_number_of_tools__(mcp_server_id)
                # Give preference to MCP server used from last prompt
                tool_id = random.randint(1, no_of_tools)
                insert_session_interaction_query = "INSERT INTO session_interactions (session_id, tool_id, sequence_number) VALUES (?, ?, ?);"
                self.dbManager.execute_query(insert_session_interaction_query, (session_id, tool_id, seq_num))
                # ----------- Compute same MCP server with some probability --------------
                # Only change mcp server if random prob is more than given
                if random.random() > CConfig.PROB_OF_TOOL_FROM_SAME_MCP: # random.random() gives [0.0, 1.0)
                    mcp_server_id = random.randint(1, CConfig.MAX_MCP_SERVERS)

    def __create_sessions_and_interactions_for_all_users__(self):
        # Show progress bar

        print("Creating sessions and interactions for all users...")
        for user_id in tqdm(range(1, CConfig.MAX_USERS + 1)):
            self.__create_sessions_and_interactions_for_user__(user_id)
          


    def __get_number_of_tools__(self, mcp_server_id):
        query = "SELECT no_of_tools FROM mcp_servers WHERE id = ?;"
        result = self.dbManager.execute_read_query(query, (mcp_server_id,))
        if result:
            return result[0][0]
        return 0

    def create_synthetic_data(self):
        dataGenerator = CGenerateSyntheticData()
        dataGenerator.__create_users__()
        dataGenerator.__create_mcp_servers_and_tools__()
        dataGenerator.__create_sessions_and_interactions_for_all_users__()

if __name__ == "__main__":
    CDatabaseManager.delete_db_file()
    dataGenerator = CGenerateSyntheticData()
    dataGenerator.create_synthetic_data()
    