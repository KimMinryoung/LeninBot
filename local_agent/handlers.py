"""Tool handler implementations for the local agent."""

import glob
import json
import logging
import os
from datetime import datetime
from functools import wraps

logger = logging.getLogger(__name__)

# ── Tool Result Size Limit ────────────────────────────────────────────
_MAX_RESULT_CHARS = 30_000  # ~10K tokens — prevents API overload


def _truncate_result(fn):
    """Decorator: truncate tool results exceeding _MAX_RESULT_CHARS."""
    @wraps(fn)
    async def wrapper(*args, **kwargs):
        result = await fn(*args, **kwargs)
        if isinstance(result, str) and len(result) > _MAX_RESULT_CHARS:
            truncated = result[:_MAX_RESULT_CHARS]
            result = truncated + f"\n\n... [truncated: {len(result):,} → {_MAX_RESULT_CHARS:,} chars]"
        return result
    return wrapper


# ── File System Tools ─────────────────────────────────────────────────

@_truncate_result
async def handle_read_file(path: str, line_start: int | None = None, line_end: int | None = None) -> str:
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    if not os.path.exists(path):
        return f"Error: File not found: {path}"
    if os.path.isdir(path):
        return f"Error: Path is a directory, not a file: {path}"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        total = len(lines)
        start = max(1, line_start or 1)
        end = min(total, line_end or total)
        selected = lines[start - 1 : end]
        numbered = [f"{start + i:>6}\t{line.rstrip()}" for i, line in enumerate(selected)]
        header = f"[{path}] lines {start}-{end} of {total}"
        return header + "\n" + "\n".join(numbered)
    except Exception as e:
        return f"Error reading file: {e}"


async def handle_write_file(path: str, content: str, mode: str = "overwrite") -> str:
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        write_mode = "a" if mode == "append" else "w"
        with open(path, write_mode, encoding="utf-8") as f:
            f.write(content)
        size = os.path.getsize(path)
        return f"Written {len(content)} chars to {path} (total size: {size} bytes, mode: {mode})"
    except Exception as e:
        return f"Error writing file: {e}"


async def handle_list_directory(path: str = ".", pattern: str = "*", recursive: bool = False) -> str:
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    if not os.path.isdir(path):
        return f"Error: Not a directory: {path}"
    try:
        if recursive:
            search = os.path.join(path, "**", pattern)
            entries = glob.glob(search, recursive=True)
        else:
            search = os.path.join(path, pattern)
            entries = glob.glob(search)
        entries.sort()
        lines = []
        for entry in entries[:200]:  # limit output
            try:
                stat = os.stat(entry)
                kind = "DIR " if os.path.isdir(entry) else "FILE"
                size = f"{stat.st_size:>10,}" if not os.path.isdir(entry) else "         -"
                mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                rel = os.path.relpath(entry, path)
                lines.append(f"  {kind} {size}  {mtime}  {rel}")
            except OSError:
                lines.append(f"  ???? {os.path.relpath(entry, path)}")
        header = f"[{path}] {len(entries)} entries"
        if len(entries) > 200:
            header += f" (showing first 200 of {len(entries)})"
        return header + "\n" + "\n".join(lines)
    except Exception as e:
        return f"Error listing directory: {e}"


# ── Web Search ────────────────────────────────────────────────────────

async def handle_web_search(query: str, max_results: int = 5) -> str:
    try:
        from tavily import TavilyClient
        client = TavilyClient()
        response = client.search(query, max_results=min(max_results, 10))
        results = response.get("results", [])
        if not results:
            return "No search results found."
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            content = r.get("content", "")[:500]
            lines.append(f"[{i}] {title}\n    URL: {url}\n    {content}\n")
        return f"Found {len(results)} results for: {query}\n\n" + "\n".join(lines)
    except Exception as e:
        return f"Web search error: {e}"


# ── Playwright Crawling ──────────────────────────────────────────────

@_truncate_result
async def handle_crawl_page(url: str, wait_for: str | None = None, extract_links: bool = False) -> str:
    try:
        from local_agent.crawler import crawl
        result = await crawl(url, wait_for=wait_for, extract_links=extract_links)
        return result
    except Exception as e:
        return f"Crawl error: {e}"


# ── Local SQLite ─────────────────────────────────────────────────────

async def handle_query_local_db(sql: str) -> str:
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT"):
        return "Error: Only SELECT queries are allowed."
    try:
        from local_agent.local_db import query
        rows = query(sql)
        if not rows:
            return "No rows returned."
        lines = [json.dumps(r, ensure_ascii=False, default=str) for r in rows[:100]]
        result = "\n".join(lines)
        if len(rows) > 100:
            result += f"\n... ({len(rows)} total rows, showing first 100)"
        return result
    except Exception as e:
        return f"SQLite error: {e}"


# ── Task Management ──────────────────────────────────────────────────

async def handle_manage_task(
    action: str,
    content: str | None = None,
    task_id: int | None = None,
    parent_task_id: int | None = None,
    status: str | None = None,
    result: str | None = None,
    scratchpad: str | None = None,
) -> str:
    from local_agent.local_db import query, execute
    try:
        if action == "add":
            if not content:
                return "Error: 'content' required for add action."
            depth = 0
            if parent_task_id:
                parent = query("SELECT depth FROM tasks WHERE id = ?", (parent_task_id,))
                if not parent:
                    return f"Error: parent task #{parent_task_id} not found."
                depth = (parent[0].get("depth") or 0) + 1
                if depth > 5:
                    return "Error: maximum task depth (5) exceeded."
            row_id = execute(
                "INSERT INTO tasks (content, parent_task_id, depth) VALUES (?, ?, ?)",
                (content, parent_task_id, depth),
            )
            chain_info = f" (subtask of #{parent_task_id}, depth={depth})" if parent_task_id else ""
            return f"Task #{row_id} created{chain_info}: {content}"

        elif action == "list":
            if status:
                rows = query("SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC LIMIT 20", (status,))
            else:
                rows = query("SELECT * FROM tasks ORDER BY created_at DESC LIMIT 20")
            if not rows:
                return "No tasks found."
            lines = []
            for r in rows:
                indent = "  " * (r.get("depth") or 0)
                parent_info = f" ←#{r['parent_task_id']}" if r.get("parent_task_id") else ""
                lines.append(f"  {indent}#{r['id']} [{r['status']}]{parent_info} {r['content'][:80]}")
                if r.get("result"):
                    lines.append(f"  {indent}     Result: {r['result'][:120]}")
                if r.get("scratchpad"):
                    lines.append(f"  {indent}     Scratchpad: {len(r['scratchpad'])} chars")
            return f"{len(rows)} task(s):\n" + "\n".join(lines)

        elif action == "update":
            if not task_id:
                return "Error: 'task_id' required for update action."
            parts, params = [], []
            if status:
                parts.append("status = ?")
                params.append(status)
                if status in ("done", "failed"):
                    parts.append("completed_at = datetime('now', 'localtime')")
            if result:
                parts.append("result = ?")
                params.append(result)
            if scratchpad is not None:
                # Enforce 20KB limit
                if len(scratchpad) > 20_000:
                    scratchpad = scratchpad[:20_000] + "\n[truncated at 20KB]"
                parts.append("scratchpad = ?")
                params.append(scratchpad)
            if not parts:
                return "Error: provide 'status', 'result', or 'scratchpad' to update."
            params.append(task_id)
            execute(f"UPDATE tasks SET {', '.join(parts)} WHERE id = ?", params)
            return f"Task #{task_id} updated."

        return f"Unknown action: {action}"
    except Exception as e:
        return f"Task management error: {e}"


# ── Server Sync ──────────────────────────────────────────────────────

async def handle_sync_push(data_type: str, content: str, metadata: dict | None = None) -> str:
    try:
        from local_agent.sync import sync_push
        return await sync_push(data_type, content, metadata or {})
    except Exception as e:
        return f"Sync push error: {e}"


async def handle_sync_pull(data_type: str, params: dict | None = None) -> str:
    try:
        from local_agent.sync import sync_pull
        return await sync_pull(data_type, params or {})
    except Exception as e:
        return f"Sync pull error: {e}"


# ── Python Code Execution ────────────────────────────────────────────

@_truncate_result
async def handle_execute_python(code: str, timeout: int = 30) -> str:
    """Write code to a temp file, execute it, return output."""
    import asyncio
    import subprocess
    import sys
    import tempfile

    timeout = max(5, min(timeout, 300))  # clamp 5-300s

    def _run():
        # Write to temp file in project root so relative imports work
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", dir=project_root,
            delete=False, encoding="utf-8",
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True, text=True, timeout=timeout,
                cwd=project_root,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            output_parts = []
            if result.stdout.strip():
                output_parts.append(result.stdout.strip())
            if result.stderr.strip():
                output_parts.append(f"[stderr]\n{result.stderr.strip()}")
            if result.returncode != 0:
                output_parts.append(f"[exit code: {result.returncode}]")
            output = "\n\n".join(output_parts) if output_parts else "(no output)"

            # Truncate if too large
            if len(output) > 50000:
                output = output[:50000] + f"\n\n... [truncated, total {len(output)} chars]"
            return output
        except subprocess.TimeoutExpired:
            return f"Error: Execution timed out after {timeout}s."
        except Exception as e:
            return f"Error: {e}"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return await asyncio.to_thread(_run)


# ── Vector DB Ingest ─────────────────────────────────────────────────

_embeddings = None

def _get_embeddings():
    """Lazy-load BGE-M3 embeddings (first call takes ~30s)."""
    global _embeddings
    if _embeddings is None:
        import torch
        from langchain_huggingface import HuggingFaceEmbeddings
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading BGE-M3 on %s (first time may take ~30s)...", device)
        _embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("BGE-M3 loaded on %s.", device)
    return _embeddings


async def handle_vectordb_ingest(
    content: str,
    source: str,
    author: str | None = None,
    year: str | None = None,
    layer: str = "modern_analysis",
    source_url: str | None = None,
) -> str:
    """Chunk text, embed with BGE-M3, and INSERT into lenin_corpus."""
    import asyncio
    try:
        def _do_ingest():
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from db import get_conn
            import json as _json

            # Strip header lines if present (Source:/Title:/Author:/Year:)
            body = content
            for prefix in ("Source:", "Title:", "Author:", "Authors:", "Year:"):
                if body.lstrip().startswith(prefix):
                    body = "\n".join(
                        line for line in body.split("\n")
                        if not line.strip().startswith(prefix)
                    )

            if len(body.strip()) < 50:
                return "Error: Content too short to ingest (< 50 chars)."

            # Chunk
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_text(body)

            # Build metadata
            metadata = {"layer": layer, "source": source}
            if author:
                metadata["author"] = author
            if year:
                metadata["year"] = year

            # Embed
            emb = _get_embeddings()
            vectors = emb.embed_documents(chunks)

            # Insert
            with get_conn() as conn:
                with conn.cursor() as cur:
                    for chunk_text, vec in zip(chunks, vectors):
                        embedding_str = "[" + ",".join(str(v) for v in vec) + "]"
                        cur.execute(
                            "INSERT INTO lenin_corpus (content, metadata, embedding) VALUES (%s, %s, %s::vector)",
                            (chunk_text, _json.dumps(metadata), embedding_str),
                        )

            return f"Ingested {len(chunks)} chunks into vectorDB (layer={layer}, source={source}, author={author}, year={year})"

        return await asyncio.to_thread(_do_ingest)
    except Exception as e:
        return f"VectorDB ingest error: {e}"


# ── Crawl Site (batch new articles) ──────────────────────────────────

@_truncate_result
async def handle_crawl_site(
    list_url: str,
    link_pattern: str,
    max_pages: int = 20,
    wait_for: str | None = None,
) -> str:
    """Crawl a listing page, discover article links, crawl only new ones."""
    try:
        from local_agent.crawler import crawl, _get_context
        from local_agent.local_db import query as db_query
        import asyncio as _aio

        # Step 1: Get listing page and extract links
        ctx = await _get_context()
        page = await ctx.new_page()
        try:
            await page.goto(list_url, wait_until="domcontentloaded", timeout=30000)
            await _aio.sleep(2)

            # For Naver Cafe, check iframe
            target_frame = page
            if "cafe.naver.com" in list_url:
                for frame in page.frames:
                    if "/ca-fe/cafes/" in frame.url or "ArticleList" in frame.url:
                        target_frame = frame
                        break

            all_links = await target_frame.evaluate("""() => {
                return Array.from(document.querySelectorAll('a[href]'))
                    .map(a => ({text: a.innerText.trim().substring(0, 200), href: a.href}))
                    .filter(l => l.href.startsWith('http') && l.text.length > 0);
            }""")
        finally:
            await page.close()

        # Step 2: Filter by pattern
        article_links = []
        seen_urls = set()
        for link in all_links:
            href = link["href"]
            if link_pattern in href and href not in seen_urls:
                seen_urls.add(href)
                article_links.append(link)

        if not article_links:
            return f"No links matching pattern '{link_pattern}' found on {list_url}"

        # Step 3: Check which are already cached
        cached = db_query("SELECT url FROM crawl_cache")
        cached_urls = {r["url"] for r in cached}
        new_links = [l for l in article_links if l["href"] not in cached_urls]

        if not new_links:
            return f"Found {len(article_links)} article links, but all {len(article_links)} are already cached. No new articles."

        # Step 4: Crawl new articles (up to max_pages)
        to_crawl = new_links[:max_pages]
        results = []
        for i, link in enumerate(to_crawl, 1):
            logger.info("Crawling %d/%d: %s", i, len(to_crawl), link["href"])
            content = await crawl(link["href"], wait_for=wait_for)
            results.append({
                "url": link["href"],
                "title": link["text"][:100],
                "content_length": len(content),
                "preview": content[:300],
            })
            if i < len(to_crawl):
                await _aio.sleep(1)  # polite delay

        # Summary
        lines = [f"Discovered {len(article_links)} total links, {len(cached_urls & seen_urls)} already cached, {len(new_links)} new."]
        lines.append(f"Crawled {len(results)} new articles:\n")
        for r in results:
            lines.append(f"  [{r['title'][:60]}] ({r['content_length']} chars)")
            lines.append(f"    URL: {r['url']}")
        lines.append(f"\nAll {len(results)} articles are now in crawl_cache. Use vectordb_ingest to add them to the vector DB.")

        return "\n".join(lines)
    except Exception as e:
        return f"Crawl site error: {e}"


# ── Handler Registry ─────────────────────────────────────────────────

LOCAL_TOOL_HANDLERS = {
    "read_file": handle_read_file,
    "write_file": handle_write_file,
    "list_directory": handle_list_directory,
    "web_search": handle_web_search,
    "crawl_page": handle_crawl_page,
    "query_local_db": handle_query_local_db,
    "manage_task": handle_manage_task,
    "sync_push": handle_sync_push,
    "sync_pull": handle_sync_pull,
    "execute_python": handle_execute_python,
    "vectordb_ingest": handle_vectordb_ingest,
    "crawl_site": handle_crawl_site,
}
