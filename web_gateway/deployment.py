"""Search-key consumers and deployment metadata shared by install tooling."""
CONSUMERS = (
    "leninbot-api", "leninbot-telegram", "novel-writer-api",
    "leninbot-roleplay", "leninbot-a2a-api", "leninbot-browser",
    "leninbot-autonomous", "leninbot-experience",
    "leninbot-worker",
)
SEARCH_KEYS = frozenset({"tavily_api_key", "brave_search_api_key"})
