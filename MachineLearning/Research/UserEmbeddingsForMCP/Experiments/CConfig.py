from dataclasses import dataclass

@dataclass(frozen=True)
class CConfig:
    def __init__(self):
        self.MAX_USERS = 10000
        self.MAX_MCP_SERVERS = 50
        self.MAX_TOOLS_PER_MCP = 20
        self.MIN_TOOLS_PER_MCP = 1
        self.SESSIONS_PER_USER_MEAN = 40
        self.SESSIONS_PER_USER_STD = 20 
        self.SESSIONS_LENGTH_MEAN = 8
        self.SESSIONS_LENGTH_STD = 4
        self.EMBEDDING_DIMENSIONS = 8
        self.DB_FILE_NAME = "mcp_interactions.db"
