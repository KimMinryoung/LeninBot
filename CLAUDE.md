# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Evolve Cyber-Lenin from a linear RAG chatbot into the most intelligent autonomous agent possible, based on CLAW (Planner + ReAct Executor + Critic triad) and other state-of-the-art agent architectures (Reflexion, LATS, Plan-and-Execute). The evolution is incremental — each phase adds a capability layer while preserving existing functionality.

## Development Documentation

- `temp_dev/project_state.md` — Current project state, architecture, capabilities, and limitations snapshot
- `temp_dev/agent_evolution_plan.md` — Full phased plan for agent evolution (Phase 0-6)
- `temp_dev/to_agent_AI.txt` — Original CLAW architecture discussion and Gemini's advice

Always consult `temp_dev/agent_evolution_plan.md` before starting new work to understand the current phase and what comes next. Update these documents as phases are completed.

## Project Overview


1. Think Before Coding
Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

State your assumptions explicitly. If uncertain, ask.
If multiple interpretations exist, present them - don't pick silently.
If a simpler approach exists, say so. Push back when warranted.
If something is unclear, stop. Name what's confusing. Ask.
2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

No features beyond what was asked.
If you write 200 lines and it could be 50, rewrite it.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

3. Surgical Changes
Touch only what you must. Clean up only your own mess.

When editing existing code:

Don't "improve" adjacent code, comments, or formatting.
Don't refactor things that aren't broken if you not asked to refactor.
When changing code, do not arbitrarily abbreviate content text or prompts, and only change them when requested.
Match existing style, even if you'd do it differently.
If you notice unrelated dead code, mention it - don't delete it.
When your changes create orphans:

Remove imports/variables/functions that YOUR changes made unused.
Don't remove pre-existing dead code unless asked.
The test: Every changed line should trace directly to the user's request.

4. Goal-Driven Execution
Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

"Add validation" → "Write tests for invalid inputs, then make them pass"
"Fix the bug" → "Write a test that reproduces it, then make it pass"
"Refactor X" → "Ensure tests pass before and after"
For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

These guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Environment
- Always activate the virtual environment before running Python commands: `source .venv/Scripts/activate`

## Caution
- When running bash commands on Windows, use /dev/null for output redirection, not nul.