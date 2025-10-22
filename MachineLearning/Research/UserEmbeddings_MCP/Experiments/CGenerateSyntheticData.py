'''
This class will create synthetic data for testing the user embeddings model.

and insert it into the database.
'''
import os,sys
import random
import numpy as np
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
        self.dbManager = CDatabaseManager(CConfig.DB_FILE_NAME)
        self.dbManager.create_tables()

    def create_users(self):
        print("Creating users...")
        for user_id in range(0, CConfig.MAX_USERS):
            insert_user_query = "INSERT INTO users (id) VALUES (?);"
            self.dbManager.execute_query(insert_user_query, (user_id,))
        print("Users created.")

    def create_mcp_servers_and_tools(self):
        print("Creating MCP servers and tools...")
        tool_id_counter = 0
        for mcp_server_id in range(0, CConfig.MAX_MCP_SERVERS):
            print(f"Creating MCP server {mcp_server_id}...")
            no_of_tools = random.randint(CConfig.MIN_TOOLS_PER_MCP, CConfig.MAX_TOOLS_PER_MCP)
            insert_mcp_query = "INSERT INTO mcp_servers (id, no_of_tools) VALUES (?, ?);"
            self.dbManager.execute_query(insert_mcp_query, (mcp_server_id, no_of_tools))
            for tool_id in range(0, no_of_tools):
                insert_tool_query = "INSERT INTO mcp_tools (id, mcp_server_id) VALUES (?, ?);"
                self.dbManager.execute_query(insert_tool_query, (tool_id_counter + tool_id, mcp_server_id))
            tool_id_counter += tool_id # This will reduce number of updates
        print("MCP servers and tools created.")

if __name__ == "__main__":
    dataGenerator = CGenerateSyntheticData()
    #dataGenerator.create_users()
    dataGenerator.create_mcp_servers_and_tools()