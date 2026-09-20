"""Search-key consumers and deployment metadata shared by install tooling."""
CONSUMERS = (
    "leninbot-api", "leninbot-telegram", "novel-writer-api",
    "leninbot-roleplay", "leninbot-a2a-api", "leninbot-browser",
    "leninbot-autonomous", "leninbot-experience",
    "leninbot-commulingo-new", "leninbot-commulingo-enrich",
    "leninbot-commulingo-maintainer", "leninbot-commulingo-terms",
    "leninbot-commulingo-gap",
    "leninbot-commulingo-review", "leninbot-commulingo-pipeline",
)
SEARCH_KEYS = frozenset({"tavily_api_key", "brave_search_api_key"})
