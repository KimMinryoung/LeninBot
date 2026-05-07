# Hot-Reloaded Agent Prompts

Files in this directory are loaded by `AgentSpec.render_prompt()` on every
prompt render. Editing them does not require restarting `leninbot-telegram` or
other long-running services.

Convention:

- `identity/agent_prompts/<agent_name>.md`
- The file is inserted as a `runtime-prompt` section.
- For agents that include `identity/political_line.md`, prompt order is:
  identity/persona, political line, runtime prompt, built-in agent sections,
  then per-turn user context such as task, project topic, project goal, plan,
  recent notes, and operator advice.

Use this directory for policy, prompt, and behavior guidance that changes often.
Do not use it for tool implementations, Python handler logic, database schema,
or command registration; those still require code changes and service restart.
