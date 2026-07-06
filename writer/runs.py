"""In-process registry of live background writer runs.

One run per project at a time. Browsers detach and reattach to a run's SSE
stream (page reload, network blip) without losing the run; tool handlers
record manuscript edit spans here so the final event can report exactly what
changed."""

from __future__ import annotations

import asyncio


class WriterRun:
    """Server-side state of one background writer run."""

    def __init__(self, *, project_id: int, run_id: str, user_message_id: int | None,
                 model: str, model_display: str) -> None:
        self.project_id = project_id
        self.run_id = run_id
        self.user_message_id = user_message_id
        self.model = model
        self.model_display = model_display
        self.subscribers: set[asyncio.Queue[str | None]] = set()
        self._text_parts: list[str] = []
        self.final_event: dict | None = None
        self.last_progress = asyncio.get_running_loop().time()
        self.edits: list[dict] = []

    def append_text(self, chunk: str) -> None:
        self._text_parts.append(chunk)

    def text_snapshot(self) -> str:
        return "".join(self._text_parts)

    def record_edit(self, action: str, start: int, end: int, delta: int) -> None:
        """Track a manuscript edit span in final-body coordinates: later edits
        that grow/shrink the body shift every previously recorded span that
        sits at or after their position."""
        if delta:
            for edit in self.edits:
                if edit["start"] >= start:
                    edit["start"] += delta
                    edit["end"] += delta
                elif edit["end"] > start:
                    # The new edit lands inside this recorded span (e.g. a
                    # critic replace within a fresh append): its end moves too.
                    edit["end"] += delta
        self.edits.append({"action": action, "start": start, "end": end})

    async def broadcast(self, item: str | None) -> None:
        stale: list[asyncio.Queue[str | None]] = []
        for queue in tuple(self.subscribers):
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                stale.append(queue)
        for queue in stale:
            self.subscribers.discard(queue)


_active_runs: dict[int, WriterRun] = {}


def get_active_run(project_id: int) -> WriterRun | None:
    return _active_runs.get(project_id)


def register_run(run: WriterRun) -> None:
    _active_runs[run.project_id] = run


def unregister_run(run: WriterRun) -> None:
    if _active_runs.get(run.project_id) is run:
        del _active_runs[run.project_id]


def record_run_edit(project_id: int, action: str, start, end, delta) -> None:
    run = _active_runs.get(project_id)
    if run is None or start is None or end is None:
        return
    run.record_edit(action, int(start), int(end), int(delta or 0))
