# MinAgent

MinAgent is a small terminal coding agent for the directory from which it is started. It connects to an OpenAI Chat Completions compatible endpoint, streams the model response as it arrives, and gives the model workspace tools for reading and changing files.

This is a Python port managed with [uv](https://docs.astral.sh/uv/), built for Linux (developed on CachyOS). It targets Python 3.12 or later.

## Requirements

- Python 3.12 or later.
- [uv](https://docs.astral.sh/uv/) on your `PATH`. Install it with `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- An OpenAI Chat Completions compatible server with SSE streaming.
- Tool calling is required for workspace file operations. Multimodal input is required only when using images.

The only third-party packages are `httpx` (streaming HTTP) and `regex` (grapheme-cluster segmentation, which the standard library cannot do).

## Getting started

```bash
git clone <this-repo> MinAgent
cd MinAgent
uv sync
cp .env.example .env
$EDITOR .env          # set OPENAI_MODEL, and the endpoint if not a local llama.cpp
```

`uv sync` creates `.venv/` and installs the dependencies plus the project itself.

## Configuration

MinAgent reads `.env` from the MinAgent project directory (the one holding `pyproject.toml`). Values already present in your environment take precedence over that file.

Example configuration for a local llama.cpp server:

```env
OPENAI_BASE_URL=http://127.0.0.1:8080/v1
OPENAI_API_KEY=llama.cpp
OPENAI_MODEL=llama.cpp
OPENAI_INPUT=text,image
OPENAI_CONTEXT_WINDOW=262144
OPENAI_TIMEOUT_SECONDS=420
MCP_TIMEOUT_SECONDS=420
TERMINAL_TIMEOUT_SECONDS=420
MAX_TOOL_ROUNDS=64
TOOL_PREVIEW_CHARS=12000
OPENAI_SHOW_REASONING=off
WORKSPACE_LIST_LIMIT=0
TERMINAL_MODE=off
SKILLS_ENABLED=off
MCP_ENABLED=off
MEMORY_ENABLED=off
WEB_SEARCH_ENABLED=off
```

`OPENAI_MODEL` is required. `OPENAI_BASE_URL` defaults to `https://api.openai.com/v1` and is normalized to the `/chat/completions` endpoint. `OPENAI_API_KEY` is optional. `OPENAI_TIMEOUT_SECONDS`, `MCP_TIMEOUT_SECONDS`, and `TERMINAL_TIMEOUT_SECONDS` are positive integers in seconds and default to `420` (seven minutes); they bound one endpoint request, one MCP request, and one shell command respectively. `MAX_TOOL_ROUNDS` is a positive integer bounding how many tool-call rounds one turn may run before stopping, and defaults to `64`. `TOOL_PREVIEW_CHARS` is a positive integer bounding the inline preview of an oversized tool result, and defaults to `12000`.

Boolean settings use only `on` and `off`:

- `OPENAI_SHOW_REASONING=on` displays the reasoning channel as muted gray text while it streams. `off` keeps the regular `Processing...` indicator. The endpoint must send `choices[0].delta.reasoning_content` (llama.cpp) or `choices[0].delta.reasoning_summary`.
- `SKILLS_ENABLED=on` loads local skills. The default is `off`.
- `MCP_ENABLED=on` loads configured MCP servers. The default is `off`.
- `MEMORY_ENABLED=on` loads MinAgent's persistent SQLite memory, which recalls what already worked and records what works. The default is `off`. `MEMORY_DB_PATH` overrides the database location; the default is `.agents/memory/memoria.db` in the MinAgent project directory. `MEMORY_DIRECT_ANSWER=off` stops MinAgent from answering a known request straight from memory; the default is `on`.
- `WEB_SEARCH_ENABLED=on` lets the model search the web and fetch pages through Ollama's hosted API. The default is `off`. It requires `OLLAMA_API_KEY` (an Ollama account key). `WEB_SEARCH_BASE_URL` defaults to `https://ollama.com/api` and `WEB_SEARCH_TIMEOUT_SECONDS` to `120`.

For llama.cpp, use `--reasoning-format deepseek` when the model template does not automatically emit a separate `reasoning_content` channel. MinAgent displays that channel as progress text and keeps the final answer in its normal response presentation.

`OPENAI_INPUT` must contain `text` and may also contain `image`. `OPENAI_CONTEXT_WINDOW` is a positive integer and defaults to `262144` tokens; set it to the actual model context limit, because this setting does not enlarge the model. When the endpoint is Ollama, MinAgent also reads the model's real runtime window from `POST /api/show` (`num_ctx`), so the effective window is the smaller of `OPENAI_CONTEXT_WINDOW` and that value even when the model name states nothing. A value above the model's real window lets the server truncate silently, and a large workspace inventory can then push the user's own question out of context. Too small a value is reported before the request instead, with the largest prompt components to trim. `WORKSPACE_LIST_LIMIT` defaults to `0`, which disables recursive inventory in the model context. `@` file autocomplete still searches a local index bounded to 10,000 entries and excludes common generated directories. The model can also call `list_directory` for a focused listing. A positive value includes up to that many entries per directory; `-1` includes all entries. Inventories stop at 10,000 entries or 128 KiB of text. `/init` builds a one-time inventory regardless of this setting. `TERMINAL_MODE` accepts lowercase `auto`, `ask`, or `off`, and defaults to `ask` when it is not set.

## Starting MinAgent

Start it from the workspace directory that the agent is allowed to modify:

```bash
cd /path/to/your/project
/path/to/MinAgent/minagent.sh
```

`minagent.sh` runs `uv run --project <MinAgent dir> minagent`, so your current directory stays the workspace while the environment comes from the MinAgent directory. It sets `MINAGENT_ROOT` so MinAgent finds its own `.env` and `.minagent/mcp.json` regardless of where you launch it from.

The workspace is the directory where the command is launched. `list_directory`, file changes, and files selected through `@` are confined to it. `read_file` and an image path explicitly written in the prompt can read one specifically named file outside it, but cannot list outside directories. Terminal commands and configured MCP servers run with your account permissions.

File paths are relative to that directory. To read a file outside it, pass its explicit absolute path or a relative path such as `../notes.txt` to `read_file`; outside directories cannot be listed and other file tools cannot change them. If MinAgent is started in `Test`, use `README.md` for `Test/README.md`. A redundant `Test/README.md` also resolves to the root file when there is no real `Test` subdirectory; if one exists, its paths take precedence. Use `./Test/file.txt` to explicitly target or create a same-named subdirectory. Without such a subdirectory, `Test` alone refers to the workspace root and cannot be read as a file or deleted.

## Conversation and streaming

The final answer streams into a shaded assistant response as tokens arrive. Press `Esc` while a model response or compaction summary is streaming to stop that request; MinAgent returns to the prompt so you can send a correction. A partial answer is kept in the conversation when available. Markdown headings, lists, code fences, links, inline formatting, and tables are rendered for the terminal. Tables are aligned to the terminal width and long cell contents wrap across lines.

When `OPENAI_SHOW_REASONING=on` and the endpoint supplies a supported reasoning delta, the reasoning is printed before the final response as muted gray text without a separate panel or background, wrapped to the terminal width so it stays aligned as a block. If the endpoint does not supply that field, MinAgent continues to show `Processing...` and the final response normally.

The model chooses when it needs workspace contents. With `WORKSPACE_LIST_LIMIT=0`, no recursive inventory is injected, while `@` file autocomplete remains available from a bounded local index. The model can call `list_directory` to inspect a specific directory's immediate entries. Otherwise, the inventory supplies paths but no file contents. MinAgent does not force an initial `read_file` call merely because files exist. When a request depends on project files, the model should call `read_file` before planning, diagnosing, or changing them. `list_directory` includes hidden entries, does not recurse, and returns at most 500 entries by default. Successful edits and writes are verified internally by MinAgent; the model does not need to read the same file back. After a failed edit, reread the file before retrying so the new edit is based on its current contents.

MinAgent verifies the persisted contents before a successful edit or write tool reports completion. A model response may request at most 16 tool calls; a turn may use at most `MAX_TOOL_ROUNDS` tool rounds (64 by default). Tool output stored in the conversation is compressed first (ANSI colour stripped, line endings normalised, blank-line and space runs collapsed, whole-JSON payloads minified) and, when it still exceeds `TOOL_PREVIEW_CHARS`, reduced to a head-and-tail preview. The omitted text is not discarded: it is stored zlib-compressed under `.minagent/tool-outputs/`, and the truncation note names an id the model passes to `recall_tool_output` to read any character range of the original back. Recall is exact, so a result is never permanently lost, and a single recall is capped so it cannot refill the window the archive just freed.

Keeping the result recoverable is what lets the inline preview stay small. Before, one result was allowed to occupy up to a quarter of the window - roughly 65,000 tokens on the default window - and whatever did not fit was gone. Now a single result costs at most `TOOL_PREVIEW_CHARS` (about 2,800 tokens by default) and the rest is one `recall_tool_output` call away. Compression earns its place here by making the off-window copy cheap to keep: an archive is capped at 64 MB and its oldest entries are pruned past that, a single result over 8 MB is not archived at all, and a session with no archive root falls back to the old truncation.

## Prompt caching and request replay

Providers only reuse a prompt prefix that is byte-identical to the one they already saw, and the system message is the first thing in the request, so a character that changes in it invalidates the whole reusable prefix. Two things keep that prefix stable. The host clock in the system prompt is truncated to the minute, because a second-resolution clock changed the system message on every single request and forced a cache miss each time; a minute of drift is irrelevant since the model can always run `date`. And the sections that legitimately change - the clock, `AGENTS.md`, the compacted summary, the workspace inventory, the memory hints - are placed after the stable ones instead of interleaved with them, so the reusable block covers as much of the prompt as it can.

On top of that, a request that is byte-identical to one already answered is replayed from a small in-memory LRU instead of being sent again. This earns nothing on the happy path, where every turn changes the prompt; it pays off on the paths that deliberately resend the same request, such as the retry after an empty or truncated response, a resubmitted prompt from the history ring, and a corrective nudge. The whole completion is replayed rather than just the text, because the caller branches on the `truncated` flag and on the usage figures. A cancelled, interrupted, or aborted response is never stored: the user is entitled to a fresh attempt. The cache holds 24 entries and is dropped on a new conversation or a model switch.

When the fixed prompt - system sections plus tool schemas - already uses at least 70% of the configured context window, startup prints a warning naming the components responsible and the settings that shrink them. If it reaches the compaction budget, the warning says the next request will be refused. Either way the cause is actionable before the first request, instead of an endpoint silently truncating the prompt.

When enabled, the inventory is refreshed before each model request. If the workspace root contains `AGENTS.md`, its content is reloaded before each request and included as project guidance up to 64 KiB, whether or not the inventory is enabled.

## Input, multiline text, and file attachments

Press `Ctrl+J` to insert a newline without sending the message. Multiline text pasted into the prompt keeps its line breaks and does not submit one request per line. Press Enter to send. Press ↑/↓ to recall previous inputs: the arrows first move between the buffer's own lines, then step back through the session's submitted inputs, and stepping past the newest entry restores what you were typing. While `@` or `/` autocomplete is open, ↑/↓ select a candidate instead.

While a response is streaming the prompt is not waiting for input, so keystrokes are ignored rather than collected into the next line; `Esc` still stops the request. A lone `Esc` is recognised after a 50 ms grace period, which is what tells it apart from the escape sequences that arrow keys and other special keys send.

Type `@` followed by a filename fragment to search workspace files. Use ↑/↓ to select a result and Enter to replace the fragment with its complete path in the current line; press Enter again to submit. Selecting a text file attaches an excerpt of up to 48 KiB. Selecting an image attaches it as multimodal input. Up to eight files and four images can be attached to one message; each file is limited to 10 MiB.

Image paths written directly in a message are detected for PNG, JPEG, GIF, and WebP files inside or outside the workspace. Outside images must be named explicitly; MinAgent does not list outside directories. It attaches the image data and removes the path from the text sent to the model. The model endpoint must support image input.

Set `NO_COLOR` to disable terminal colors.

## Commands

Type `/` to open command autocomplete. Use ↑/↓ to choose a command and Enter to complete it in the current line; press Enter again to run it. The available commands are:

- `/context`: show approximate token counts for system sections, available tool schemas, and conversation history, plus the latest endpoint-reported `prompt_tokens` when available.
- `/compact [instructions]`: summarize history older than the recent ~20,000-token window. Compaction cuts only at safe user or completed assistant-message boundaries, so a large completed tool round can be summarized as a unit.
- `/init [focus]`: inspect a one-time workspace inventory and selected project files, show which files were selected, and create or update the workspace root `AGENTS.md`. It reads up to 24 files, with excerpt and total-size limits.
- `/skills [reload | show <name> | delete <name>]`: list the registered skills, force a rescan, print one skill's instructions, or delete a skill that lives inside the workspace.
- `/memory [forget <id>]`: show how many memories exist, their success and reuse counts, and the strongest entries; `forget` deletes one entry.
- `/skill <what it should do>`: ask the model to draft a `SKILL.md` for that capability, register it immediately, and report the resulting name and path. An unfinished draft is reported; nothing is registered unless it validates.
- `/model [name]`: list the models the endpoint advertises through its OpenAI-compatible `/models` endpoint (Ollama and llama.cpp both expose it), marking the current one, or switch to `name` when given. While you type `/model `, ↑/↓ choose from a live picker and Enter completes the name. Switching persists `OPENAI_MODEL` in the project `.env` and warms the model with a one-token request so the first turn is not the load.
- `/doctor`: check the model, the context window, and the fixed prompt overhead.
- `/new`: clear the screen and start a new conversation.
- `/exit`: close MinAgent.

Compaction also runs automatically as the usable context window fills. The usable window is the smaller of `OPENAI_CONTEXT_WINDOW` and any size stated by the model name (for example `...-8k`), so a model whose name states a smaller window compacts before the server truncates the prompt. The summary preserves file paths, decisions, unresolved work, user preferences, and verification state. It reduces conversation history; the system prompt, workspace guidance, inventory, and tool schemas remain. `/compact` reports both history and total context before and after, and `/context` shows the fixed prompt and tool-schema estimates.

## Workspace tools

The model can use these built-in tools; directory listings and file changes stay within the workspace root:

- `read_file`: read a specifically named UTF-8 text file inside or outside the workspace, or a supported image when image input is enabled. It cannot list directories. Text output is limited to 300 lines and 48 KiB. For a long line, use the returned `offset` and `column` to continue within that line.
- `list_directory`: list immediate files and subdirectories, including hidden entries, without recursion. It defaults to the workspace root and 500 entries; pass a workspace-relative `path` or a larger `limit` when needed. Output is capped at 50 KiB and 10,000 entries; symbolic links are shown but never followed.
- `edit_file`: replace one exact, unique text block in an existing file.
- `write_file`: create or atomically replace one UTF-8 file, creating its missing parent directories. It writes a file, never a folder: a path ending in `/` is refused.
- `create_directory`: create a folder and any missing parent folders; an existing folder is reported as such.
- `delete_file`: delete one regular file.
- `delete_directory`: recursively delete a regular subdirectory after validating its contents.
- `run_terminal`: available only when `TERMINAL_MODE` is `auto` or `ask`. It runs in the workspace directory; `ask` requires approval for each command. It is the tool for system facts such as the current date and time, the environment, or installed tools.
- `recall`, `remember`, and `record_outcome`: available only when `MEMORY_ENABLED` is `on`. `recall` searches the persistent memory before a task, `remember` saves a verified procedure or conclusion, and `record_outcome` reinforces or degrades a memory after it is reused.
- `web_search` and `web_fetch`: available only when `WEB_SEARCH_ENABLED` is `on`. `web_search` returns titles, URLs, and snippets; `web_fetch` reads one result page. Web content is untrusted data.

The prompt carries the host's local date and time, refreshed with every request, so a time question is answered from the real clock instead of a guess. When `TERMINAL_MODE` is not `off`, the prompt also states that `run_terminal` can read the rest of the system. If a reply claims a capability is unavailable without calling any tool, MinAgent sends one corrective message listing the tools that are actually available and asks the model to use one, or to save a reusable skill with `write_skill` when something is genuinely missing, rather than ending the turn on "I have no access". A reply that only describes the next step ("voy a listar los correos") without calling a tool gets the same single nudge, so an announced plan is not mistaken for the work. A second refusal is returned as the answer, so the turn never loops over it.

Reads check file identity and changes around opening and reading. `read_file` can read only a specifically named outside file; outside directories cannot be discovered through `list_directory`, and edit, write, and delete tools remain confined to the workspace. Within the workspace, file operations check for symbolic links, hard links, special files, and paths outside the workspace. Individual reads and writes are limited to 10 MiB. The workspace root cannot be deleted. A successful edit or write is reread and compared with the requested content before the tool reports success. As with all path-based file operations, an untrusted process that concurrently swaps parent directories can still race a rename or deletion; use a workspace directory tree that other untrusted processes cannot modify.

## Skills

When `SKILLS_ENABLED=on`, MinAgent discovers `SKILL.md` files in these directories:

- MinAgent `skills/<skill-name>/`
- MinAgent `.agents/skills/<skill-name>/`
- Workspace `skills/<skill-name>/`
- Workspace `.agents/skills/<skill-name>/`

Each manifest requires YAML frontmatter with `name` and `description`. The first 24 valid skills are loaded; catalog descriptions are shortened to 160 characters each and 8 KiB total. A skill file is limited to 64 KiB and a supporting resource to 32 KiB. The model uses two skill tools: `load_skill` (omit `path` to load instructions, or set it to read a bundled resource) and `write_skill`, which authors a new skill under the workspace's `.agents/skills/<name>/SKILL.md`.

Skill names are slugified (`Notas de versión` becomes `notas-de-version`) so the folder and the frontmatter always agree.

The skill directories are rescanned before each model request, using a cheap fingerprint of the folders and their `SKILL.md` files, so a skill written by hand, by the model, by `/skill`, or by another tool is registered during the same session without a restart. A newly registered skill is announced in the terminal, appears in the system prompt catalogue, and becomes loadable immediately; deleting a `SKILL.md` withdraws it just as quickly. Skills are disabled by default.

## MCP servers

When `MCP_ENABLED=on`, MinAgent reads `.minagent/mcp.json` from the MinAgent project directory. The file must contain an `mcpServers` object. Servers can use local stdio transport or Streamable HTTP:

```json
{
  "mcpServers": {
    "project-tools": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/tool", "my-mcp-server"],
      "cwd": "/path/to/your/project"
    },
    "remote-tools": {
      "url": "http://127.0.0.1:3000/mcp",
      "headers": {}
    }
  }
}
```

MinAgent reads the file at startup and re-reads it before each request when the file changed, reconnecting the servers and withdrawing the tools of a server that is no longer configured. Servers authored by the model through `write_mcp_server` are written to the project's `.agents/mcp/<name>/` directory and registered in the same `.minagent/mcp.json`. It exposes up to 32 tools to the model. Each input schema is limited to 8 KiB, all exposed tool definitions together to 64 KiB, and combined server instructions to 8 KiB. MCP text results are limited to 48,000 characters; supported images follow the same 10 MiB and four-image limits as local attachments. MCP servers run with your account permissions.

## Memory

When `MEMORY_ENABLED=on`, MinAgent keeps a local SQLite database of what it has learned at `.agents/memory/memoria.db` (override with `MEMORY_DB_PATH`). The database holds entries with a kind (`procedure`, `solution`, `fact`, `preference`, or `experience`), a title, the content, tags, and running success, failure, and reuse counts. Duplicate titles reinforce the existing entry instead of creating a second one, and the weakest entries are pruned once the store grows past its cap. The store runs in WAL mode, keeps indexes for confidence, recency, and last use, and on open refreshes the query planner with `PRAGMA optimize` and vacuums the file when free pages dominate.

Before each request, the most relevant entries are recalled and appended to the prompt as a short `Memory hints` section, so a known procedure is reused instead of rediscovered. The model can also call `recall` to search on demand. The fixed prompt overhead warning and `/context` count that hint block like any other section; set `MEMORY_ENABLED=off` to drop it.

To save a whole model round-trip, MinAgent first asks the store whether it already knows the request: a memory that matches at least 60% of the request's tokens and holds confidence of at least 0.75 is streamed back as the answer without contacting the endpoint. Anything weaker still reaches the model with hints. Set `MEMORY_DIRECT_ANSWER=off` to always consult the model.

Learning is recursive: after a turn that used tools and finished without a tool error, MinAgent stores the request, the tools used with the concrete arguments that ran (for example `run_terminal(command=uv run pytest -q)`), and the outcome as an `experience` entry unless the model already saved one with `remember`. A later session therefore starts from how the work was actually done, not just that it was. The model calls `remember` for the concrete procedure after a verified success and `record_outcome` to reinforce or correct a recalled memory, so confidence reflects what actually works across sessions.

## Web search

When `WEB_SEARCH_ENABLED=on` and `OLLAMA_API_KEY` is set, the model can reach the web through Ollama's hosted API. It is told to call `web_search` when it does not know how to do something, when a task has already failed three or more times, or when it needs current information, and `web_fetch` to read one result page in full. MinAgent also nudges the model toward `web_search` on its own after three tool errors in one turn, instead of letting it retry the same approach.

Web results are untrusted data: the tool descriptions and the prompt say so, and fetched content is bounded to 24 KiB per call. Requests go to `WEB_SEARCH_BASE_URL` (default `https://ollama.com/api`) with your `OLLAMA_API_KEY` and time out after `WEB_SEARCH_TIMEOUT_SECONDS` (default 120).

## Project layout

- `src/minagent/app.py`: TUI, conversation loop, tool dispatch, and commands.
- `src/minagent/attachments.py` and `src/minagent/image.py`: file attachments and image handling.
- `src/minagent/markdown_terminal.py` and `src/minagent/terminal_text.py`: streaming Markdown and terminal text layout.
- `src/minagent/line_editor.py` and `src/minagent/editor.py`: the raw-mode prompt, autocomplete, and key handling.
- `src/minagent/terminal_command.py` and `src/minagent/processes.py`: terminal execution and process cleanup.
- `src/minagent/openai.py`: OpenAI-compatible SSE client, configurable timeout, tool-call reassembly, and reasoning deltas.
- `src/minagent/config.py`: `.env` loading and configuration validation.
- `src/minagent/workspace.py`: workspace boundaries and file operations.
- `src/minagent/context.py`: token estimation, conversation serialization, and compaction.
- `src/minagent/skills.py`: local skill discovery and skill tools.
- `src/minagent/mcp.py`: MCP configuration, transports, tool discovery, and result handling.
- `src/minagent/init_project.py`: project file selection for `/init`.
- `src/minagent/memory.py`: the persistent SQLite memory, its recall/remember tools, and the prompt hints.
- `src/minagent/web_search.py`: the Ollama-backed `web_search` and `web_fetch` tools.
- `.agents/skills/` and `.agents/mcp/`: local skills and local MCP servers that ship with MinAgent.
- `minagent.sh`: the uv launcher.

## Tests

```bash
uv run pytest
```

The tests use `pytest` and `pytest-asyncio` and cover workspace files and their safety checks, attachments, context chunking, streaming responses, tool rounds, skill discovery and authoring, the injected host clock, the startup prompt-overhead warning, and no-access recovery, terminal approval and timeouts, the editor, terminal rendering, a local MCP HTTP server, a streaming endpoint that answers a clock question by running `date`, the persistent memory store, its tools, its prompt hints, and its automatic experience capture, and the Ollama-backed web search and fetch client against a mock HTTP transport.

## License and notice

MinAgent's own code is licensed under the [MIT License](LICENSE). See [NOTICE.md](NOTICE.md) for the Pi attribution. The project is a standalone implementation inspired by the Pi agent harness; it does not include Pi source files.
