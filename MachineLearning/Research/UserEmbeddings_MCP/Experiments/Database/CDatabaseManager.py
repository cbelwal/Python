import os,sys

# Ensure the root folder path is in sys.path ---
topRootPath = os.path.dirname(
              os.path.dirname(
              os.path.dirname(
              os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(topRootPath)
#----------------------------------------------

from UserEmbeddings_MCP.Experiments.CConfig import CConfig
from UserEmbeddings_MCP.Experiments.Database.CSQLLite import CSQLLite

class CDatabaseManager:
    def __init__(self):
        self.sqlLite = CSQLLite(CDatabaseManager.get_database_file_path())

    @staticmethod
    def get_database_file_path() -> str:
        folderPath =    os.path.dirname(
                        os.path.dirname( #Experiments
                        #os.path.dirname( #UserEmbeddings
                        os.path.abspath(__file__)))
        dbFolder = os.path.join(folderPath, "Data")
        dbFilePath = os.path.join(dbFolder, CConfig.DB_FILE_NAME)
        return dbFilePath
    
    @staticmethod
    def delete_db_file():
        db_file_path = CDatabaseManager.get_database_file_path()
        if os.path.exists(db_file_path):
            os.remove(db_file_path)
            print(f"Database file {db_file_path} deleted.")
        else:
            print(f"Database file {db_file_path} does not exist.")

    '''
    MCPServers: id, no_of_tools
    Tools: id, mcp_id
    Users: id 
    Sessions: id, user_id, session_depth
    SessionDetails: id, session_id, tool_id,  sequence_number
    '''
    def create_tables(self):
        # Create tables if they do not exist
        create_servers_table = """
        CREATE TABLE IF NOT EXISTS mcp_servers (
            id INTEGER PRIMARY KEY,
            no_of_tools INTEGER
        );
        """

        create_tools_table = """
        CREATE TABLE IF NOT EXISTS mcp_tools (
            id INTEGER PRIMARY KEY,
            mcp_server_id INTEGER,
            mcp_tool_id INTEGER,
            FOREIGN KEY (mcp_server_id) REFERENCES mcp_servers (id)
        );
        """
        
        create_users_table = """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY
        );
        """

        create_sessions_table = """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            session_depth INTEGER,
            FOREIGN KEY (user_id) REFERENCES users (id)
        );
        """

        create_session_interactions_table = """
            CREATE TABLE IF NOT EXISTS session_interactions (
            id INTEGER PRIMARY KEY,
            session_id INTEGER,
            tool_id INTEGER,
            sequence_number INTEGER,
            FOREIGN KEY (session_id) REFERENCES users (id)
            FOREIGN KEY (tool_id) REFERENCES mcp_tools (id)
        );
        """
        
        self.sqlLite.execute_query(create_servers_table)
        self.sqlLite.execute_query(create_tools_table)
        self.sqlLite.execute_query(create_users_table)
        self.sqlLite.execute_query(create_sessions_table)
        self.sqlLite.execute_query(create_session_interactions_table)

    def last_insert_rowid(self):
        (id,) = self.dbManager.execute_read_query("SELECT last_insert_rowid();")[0]
        return id

    def execute_query(self, query, params=()):
        return self.sqlLite.execute_query(query, params)
    
    def execute_read_query(self, query, params=()):
        return self.sqlLite.execute_read_query(query, params)

    def delete_all_data(self):
        delete_session_interactions = "DELETE FROM session_interactions;"
        delete_sessions = "DELETE FROM sessions;"
        delete_users = "DELETE FROM users;"
        delete_tools = "DELETE FROM mcp_tools;"
        delete_servers = "DELETE FROM mcp_servers;"
        
        self.sqlLite.execute_query(delete_session_interactions)
        self.sqlLite.execute_query(delete_sessions)
        self.sqlLite.execute_query(delete_users)
        self.sqlLite.execute_query(delete_tools)
        self.sqlLite.execute_query(delete_servers)
    
if __name__ == "__main__": # For testing purposes
    CDatabaseManager.delete_db_file()
    dbManager = CDatabaseManager()
    dbManager.create_tables()