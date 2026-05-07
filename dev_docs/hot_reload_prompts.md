# Hot-Reloaded Prompt Policy

`AgentSpec.render_prompt()` loads two classes of prompt text at render time:

- `identity/political_line.md`
- `identity/agent_prompts/<agent_name>.md`

This means prompt policy edits are visible on the next LLM call without service
restart. Python modules are still imported normally and remain cached in
`sys.modules`; changing tool code, Telegram handlers, DB logic, or agent registry
code still requires restarting the owning service.

Prompt order for IR-based agents:

1. Built-in identity/persona from Python
2. `identity/political_line.md`, if `include_political_line=True`
3. `identity/agent_prompts/<agent_name>.md`, if present
4. Built-in agent sections from Python
5. Per-turn user message context, such as task, project topic, project goal,
   current plan, recent notes, and operator advice

Use hot-reload prompt files for frequently tuned operating policy. Keep stable
role identity, tool allowlists, and implementation logic in Python.
