from dataclasses import dataclass

@dataclass(frozen=True)
class CConfig:
    MAX_USERS = 10000
    MAX_MCP_SERVERS = 50
    MAX_TOOLS_PER_MCP = 20
    MIN_TOOLS_PER_MCP = 1
    SESSIONS_PER_USER_MEAN = 40
    SESSIONS_PER_USER_STD = 20 
    SESSIONS_LENGTH_MEAN = 8
    SESSIONS_LENGTH_STD = 4
    EMBEDDING_DIMENSIONS = 8
    DB_FILE_NAME = "mcp_interactions.db"
