from dataclasses import dataclass

@dataclass(frozen=True)
class CConfig:
    MAX_USERS = 1#000 #100000
    MAX_MCP_SERVERS = 100
    MAX_TOOLS_PER_MCP = 50
    MIN_TOOLS_PER_MCP = 1
    SESSIONS_PER_USER_MEAN = 100
    SESSIONS_PER_USER_STD = 60 
    SESSIONS_LENGTH_MEAN = 20
    SESSIONS_LENGTH_STD = 10
    EMBEDDING_DIMENSIONS = 8
    PROB_OF_TOOL_FROM_SAME_MCP = 0.33
    DB_FILE_NAME = "mcp_interactions.db"
