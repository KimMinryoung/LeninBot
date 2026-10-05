"""Idempotent editorial pipeline calls through the frontend's admin MCP, with an explicit receipt for every mutation."""
from commulingo.mcp_client import call_tool, read_entry


def call(request):
    request = dict(request)
    command = request.pop('command', None)
    target = request.pop('target', None)
    if command == 'read':
        return read_entry(target, request.get('id'))
    return call_tool('editorial_pipeline', {'command': command, 'target': target, 'request': request})
