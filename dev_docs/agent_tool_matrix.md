# Agent Tool Matrix

최종 확인 기준: 현재 코드의 `agents.list_agents()` 런타임 스펙.

이 표는 각 specialist AgentSpec이 직접 실행할 수 있는 도구 목록이다. Telegram orchestrator, public web chat, MCP gateway는 별도 allow-list를 사용하므로 이 표와 동일하지 않다.

| Agent | Role | Tools | Finalization tools | Terminal tools |
|---|---|---|---|---|
| programmer | Code writing, modification, debugging, and file editing specialist (delegated to Codex CLI) | list_agent_tools | - | - |
| scout | External platform reconnaissance, community monitoring, web patrol specialist | moltbook, mersoom, web_search, fetch_url, fetch_x_post, check_inbox, allowlist_sender, download_image, download_file, convert_document, read_file, search_files, write_file, list_directory, read_self, write_kg_structured, save_finding, mission, upload_to_r2 | - | - |
| visualizer | Image prompt design and visual concept specialist based on Soviet constructivist/Rodchenko aesthetics | generate_image, read_file, write_file, fetch_url, download_image, web_search, read_self, save_finding, mission, upload_to_r2 | - | - |
| analyst | Information analysis, KG cross-validation, trend/pattern extraction, knowledge gap identification specialist | knowledge_graph_search, vector_search, web_search, fetch_url, fetch_x_post, download_file, convert_document, read_file, search_files, list_directory, read_self, write_kg_structured, save_finding, mission, research_document, get_finance_data, edit_content | - | - |
| browser | AI browser automation — login, form filling, multi-page navigation, dynamic site data extraction | browse_web, web_search, fetch_url, fetch_x_post, check_inbox, allowlist_sender, write_file, list_directory, read_file, read_self, write_kg_structured, save_finding, mission, upload_to_r2 | - | - |
| diary | Periodic diary writer — generates reflective diary entries from recent activity, news, and knowledge | read_self, recall_experience, web_search, fetch_url, fetch_x_post, knowledge_graph_search, write_kg_structured, get_finance_data, save_diary, edit_content | - | save_diary |
| stasova | Publication security (OpSec) reviewer for public-bound writing; flags risks and alternative wording without veto power | read_file, write_file, fetch_url | - | - |
| diplomat | External communications diplomat: A2A agent-to-agent messaging, email sending/receiving, inter-agent coordination | a2a_send, send_email, check_inbox, allowlist_sender, web_search, fetch_url, read_self, save_finding, mission | - | - |
| autonomous_project | Scheduled autonomous agent — advances one long-term project per hourly wake. Research + publishing to cyber-lenin.com. | web_search, fetch_url, fetch_x_post, vector_search, knowledge_graph_search, read_self, recall_experience, get_finance_data, download_file, convert_document, read_document, write_kg_structured, research_document, publish_hub_curation, edit_content, publish_static_page, add_research_note, read_research_notes, revise_plan, set_project_state | add_research_note, revise_plan, set_project_state, research_document, publish_hub_curation, edit_content, publish_static_page | - |

Notes:

- Empty `finalization_tools`/`terminal_tools` means the agent uses only its normal tool list.
- `finalization_tools` remain callable after budget/round exhaustion so durable state can still be saved.
- `terminal_tools` end the loop after a successful call and use the tool result as the task report.
