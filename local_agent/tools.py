"""Tool definitions for the local agent (Anthropic API format)."""

LOCAL_TOOLS = [
    {
        "name": "read_file",
        "description": "Read a local file and return its content with line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path (absolute or relative to working dir)."},
                "line_start": {"type": "integer", "description": "Start line (1-based, inclusive). Omit to read from beginning."},
                "line_end": {"type": "integer", "description": "End line (inclusive). Omit to read to end."},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a local file. Creates parent directories if needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write."},
                "content": {"type": "string", "description": "Content to write."},
                "mode": {"type": "string", "enum": ["overwrite", "append"], "description": "Write mode. Default: overwrite."},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_directory",
        "description": "List files and directories. Supports glob patterns.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path. Default: current directory."},
                "pattern": {"type": "string", "description": "Glob pattern filter. Default: * (all)."},
                "recursive": {"type": "boolean", "description": "Search recursively. Default: false."},
            },
            "required": [],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web via Tavily. Returns titles, URLs, and content snippets.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "max_results": {"type": "integer", "description": "Number of results (1-10). Default: 5."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "crawl_page",
        "description": "Crawl a web page with Playwright (handles JS-rendered content, login sessions preserved). Returns extracted text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to crawl."},
                "wait_for": {"type": "string", "description": "CSS selector to wait for before extraction."},
                "extract_links": {"type": "boolean", "description": "Also extract all links from the page. Default: false."},
            },
            "required": ["url"],
        },
    },
    {
        "name": "query_local_db",
        "description": "Query the local SQLite database (tasks, crawl_cache, conversations). Read-only.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "SQL SELECT query."},
            },
            "required": ["sql"],
        },
    },
    {
        "name": "manage_task",
        "description": "Manage local task queue. Actions: add (create task), list (show tasks), update (change status/result).",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "list", "update"], "description": "Action to perform."},
                "content": {"type": "string", "description": "Task description (for 'add')."},
                "task_id": {"type": "integer", "description": "Task ID (for 'update')."},
                "parent_task_id": {"type": "integer", "description": "Parent task ID to chain from (for 'add'). Creates a subtask."},
                "status": {"type": "string", "enum": ["pending", "running", "done", "failed"], "description": "New status (for 'update' or filter for 'list')."},
                "result": {"type": "string", "description": "Task result text (for 'update')."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "sync_push",
        "description": "Push data to the central server. Types: kg_episode (write to Knowledge Graph), report (save as completed task).",
        "input_schema": {
            "type": "object",
            "properties": {
                "data_type": {"type": "string", "enum": ["kg_episode", "report"], "description": "Type of data to push."},
                "content": {"type": "string", "description": "Content to push."},
                "metadata": {
                    "type": "object",
                    "description": "Additional metadata. For kg_episode: name, group_id, source_type. For report: priority.",
                },
            },
            "required": ["data_type", "content"],
        },
    },
    {
        "name": "vectordb_ingest",
        "description": "Ingest text into the central vector DB (pgvector lenin_corpus). Chunks the text, embeds with BGE-M3, and inserts. Use after crawling articles to make them searchable by the chatbot.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Full article text to ingest."},
                "source": {"type": "string", "description": "Document title or source name (shown in search results)."},
                "author": {"type": "string", "description": "Author or organization name."},
                "year": {"type": "string", "description": "Publication year (e.g. '2026')."},
                "layer": {
                    "type": "string",
                    "enum": ["core_theory", "modern_analysis"],
                    "description": "Knowledge layer. Default: modern_analysis.",
                },
                "source_url": {"type": "string", "description": "Original URL of the article (for dedup tracking)."},
            },
            "required": ["content", "source"],
        },
    },
    {
        "name": "crawl_site",
        "description": "Crawl a site's listing page to discover article links, then crawl only NEW articles (not in crawl_cache). Returns list of crawled articles with their content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "list_url": {"type": "string", "description": "URL of the listing/index page containing article links."},
                "link_pattern": {"type": "string", "description": "Substring that article URLs must contain to be selected (e.g. '/864', 'capybaralib/'). Filters out navigation/external links."},
                "max_pages": {"type": "integer", "description": "Maximum number of NEW articles to crawl. Default: 20."},
                "wait_for": {"type": "string", "description": "CSS selector to wait for on each article page."},
            },
            "required": ["list_url", "link_pattern"],
        },
    },
    {
        "name": "execute_python",
        "description": "Write and execute Python code on the fly. Creates a temp file, runs it, and returns stdout/stderr. Use for data processing, calculations, file manipulation, analysis, or any task that benefits from code. The code runs in the same Python environment (has access to installed packages, project files, etc.).",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute."},
                "timeout": {"type": "integer", "description": "Max execution time in seconds. Default: 30."},
            },
            "required": ["code"],
        },
    },
    {
        "name": "mission",
        "description": "Manage mission context — a shared timeline between chat and tasks. Actions: status (view active mission + recent events), add_event (record finding/decision/tool_result), list_events (full timeline), close (end mission), create (manually start a new mission).",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "add_event", "list_events", "close", "create"],
                    "description": "Action to perform.",
                },
                "title": {"type": "string", "description": "Mission title (for 'create')."},
                "content": {"type": "string", "description": "Event content (for 'add_event')."},
                "event_type": {
                    "type": "string",
                    "enum": ["finding", "decision", "tool_result"],
                    "description": "Event type (for 'add_event'). Default: finding.",
                },
                "source": {"type": "string", "description": "Event source (for 'add_event'). Default: chat."},
                "limit": {"type": "integer", "description": "Max events to return (for 'list_events'). Default: 50."},
            },
            "required": ["action"],
        },
    },
    {
        "name": "sync_pull",
        "description": "Pull data from the central server. Types: diaries, chat_logs, task_reports, kg_stats, experience.",
        "input_schema": {
            "type": "object",
            "properties": {
                "data_type": {
                    "type": "string",
                    "enum": ["diaries", "chat_logs", "task_reports", "kg_stats", "experience"],
                    "description": "Type of data to pull.",
                },
                "params": {
                    "type": "object",
                    "description": "Optional params. diaries: limit, keyword. chat_logs: limit, hours_back, keyword. task_reports: limit, status. experience: query, k.",
                },
            },
            "required": ["data_type"],
        },
    },
]
