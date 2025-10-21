import os,sys
from CSQLLite import CSQLLite
from Experiments.CConfig import CConfig

class CDatabaseManager:
    def __init__(self,fileName=None):
        self.fileName = fileName
        self.sqlLite = CSQLLite(self.get_database_file_path())


    @staticmethod
    def get_database_file_path(self):
        folderPath =    os.path.dirname(
                        os.path.dirname( #Experiments
                        #os.path.dirname( #UserEmbeddings
                        os.path.abspath(__file__)))
        dbFolder = os.path.join(folderPath, "Data")
        dbFilePath = os.path.join(dbFolder, self.fileName if self.fileName else "stocks_database.sqlite")
        return dbFilePath
    
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
            no_of_tools INTEGER,
        );
        """

        create_tools_table = """
        CREATE TABLE IF NOT EXISTS mcp_tools (
            id INTEGER PRIMARY KEY,
            mcp_id INTEGER,
            FOREIGN KEY (mcp_id) REFERENCES mcp_servers (id)
        );
        """
        
        create_users_table = """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
        );
        """

        create_sessions_table = """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            session_depth INTEGER,
            FOREIGN KEY (user_id) REFERENCES users (id),
        );
        """

        create_session_details_table = """
            CREATE TABLE IF NOT EXISTS session_details (
            id INTEGER PRIMARY KEY,
            session_id INTEGER,
            tool_id INTEGER,
            sequence_number INTEGER,
            FOREIGN KEY (session_id) REFERENCES users (id)
            FOREIGN KEY (tool_id) REFERENCES mcp_tools (id),
        );
        """
        
        self.sqlLite.execute_query(create_servers_table)
        self.sqlLite.execute_query(create_tools_table)
        self.sqlLite.execute_query(create_users_table)
        self.sqlLite.execute_query(create_sessions_table)
        self.sqlLite.execute_query(create_session_details_table)

    def add_database(self, name, database):
        self.databases[name] = database

    def get_database(self, name):
        return self.databases.get(name)

    def remove_database(self, name):
        if name in self.databases:
            del self.databases[name]

    def list_databases(self):
        return list(self.databases.keys())
    
if __name__ == "__main__": # For testing purposes
    dbManager = CDatabaseManager(CConfig.DB_FILE_NAME)
    dbManager.create_tables()