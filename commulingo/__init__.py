"""CommuLingo dictionary domain on leninbot's side: the curator tools
(`commulingo.people`) and the author/review/discovery agent sessions
(`commulingo.pipeline`) the agent worker runs for the frontend pipeline. The
frontend repository owns the data, the queue and rendering; leninbot reads and
writes only through its admin MCP (`commulingo.mcp_client`, `commulingo.reads`)."""
