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
TOOL_RESULT_KEEP=3
PARALLEL_TOOLS=on
CAPABILITY_IDLE_TURNS=2
CONTEXT_HIGH_WATERMARK=75%
CONTEXT_LOW_WATERMARK=55%
OPENAI_SHOW_REASONING=off
WORKSPACE_LIST_LIMIT=0
TERMINAL_MODE=off
SKILLS_ENABLED=off
MCP_ENABLED=off
MCP_APPROVAL_MODE=ask
MEMORY_ENABLED=off
WEB_SEARCH_ENABLED=off
```

`OPENAI_MODEL` is required. `OPENAI_BASE_URL` defaults to `https://api.openai.com/v1` and is normalized to the `/chat/completions` endpoint. `OPENAI_API_KEY` is optional. `OPENAI_TIMEOUT_SECONDS`, `MCP_TIMEOUT_SECONDS`, and `TERMINAL_TIMEOUT_SECONDS` are positive integers in seconds and default to `420` (seven minutes); they bound one endpoint request, one MCP request, and one shell command respectively. `MAX_TOOL_ROUNDS` is a positive integer bounding how many tool-call rounds one turn may run before stopping, and defaults to `64`. `TOOL_PREVIEW_CHARS` is a positive integer bounding the inline preview of an oversized tool result, and defaults to `12000`. `TOOL_RESULT_KEEP` is a positive integer bounding how many recent tool results stay verbatim in the transcript, and defaults to `3`. `CAPABILITY_IDLE_TURNS` is how many turns a loaded capability survives unused before its tools leave the prompt again, and defaults to `2`; `0` keeps everything loaded for the whole conversation. `CONTEXT_HIGH_WATERMARK` and `CONTEXT_LOW_WATERMARK` are the shares of the window at which the context governor starts giving things up and gives them back, and default to `75%` and `55%`; both accept `75%`, `0.75`, or `75`, and the low mark must leave room below the high one.

The last four are grouped because they are the capabilities that reach outside the workspace: the desktop, the camera and microphone, the model server, and the agent's own code. All four are off unless a `.env` turns them on, and each is gated again at the point of use, so a session that did not enable one refuses the call rather than working quietly.

Boolean settings use only `on` and `off`:

- `OPENAI_SHOW_REASONING=on` displays the reasoning channel as muted gray text while it streams. `off` keeps the regular `Processing...` indicator. The endpoint may send that channel as `choices[0].delta.reasoning_content` (llama.cpp), `choices[0].delta.reasoning_summary` (some gateways) or `choices[0].delta.reasoning` (Ollama's OpenAI-compatible surface); all three are read. This is not cosmetic: a thinking model that answers only in that channel would otherwise look like an endpoint that returned nothing.
- `INPUT_ENABLED=on` lets the agent move the pointer, click, scroll, and type. See [Desktop control](#desktop-control-keyboard-and-mouse). Off by default.
- `SENSES_ENABLED=on` lets the agent use the webcam and microphone, on request only. See [Camera and microphone](#camera-and-microphone). Off by default.
- `OLLAMA_MODELS_ENABLED=on` lets the agent manage the local Ollama models. See [Managing local models](#managing-local-models). Off by default.
- `OLLAMA_PUSH_ENABLED=on` additionally lets the agent publish a model to a registry. The default is `off`, and even on, the client refuses public ollama.com names and every push is confirmed in the terminal. Off by default.
- `SUBAGENTS_ENABLED=on` lets the agent write its own capability modules. Durable ones need terminal approval. See [Writing its own capabilities](#writing-its-own-capabilities). Off by default.
- `SKILLS_ENABLED=on` loads local skills. The default is `off`.
- `MCP_ENABLED=on` loads configured MCP servers. The default is `off`.
- `MEMORY_ENABLED=on` loads MinAgent's persistent SQLite memory, which recalls what already worked and records what works. The default is `off`. `MEMORY_DB_PATH` overrides the database location; the default is `.agents/memory/memoria.db` in the MinAgent project directory. `MEMORY_DIRECT_ANSWER=off` stops MinAgent from answering a known request straight from memory; the default is `on`. `MEMORY_EMBED_MODEL=nomic-embed-text` also compares stored memories by meaning, not only by shared words, and is `off` when empty.
- `WEB_SEARCH_ENABLED=on` lets the model search the web and fetch pages through Ollama's hosted API. The default is `off`. It requires `OLLAMA_API_KEY` (an Ollama account key). `WEB_SEARCH_BASE_URL` defaults to `https://ollama.com/api` and `WEB_SEARCH_TIMEOUT_SECONDS` to `120`.

For llama.cpp, use `--reasoning-format deepseek` when the model template does not automatically emit a separate `reasoning_content` channel. MinAgent displays that channel as progress text and keeps the final answer in its normal response presentation.

`OPENAI_INPUT` must contain `text` and may also contain `image`. `OPENAI_CONTEXT_WINDOW` is a positive integer and defaults to `262144` tokens; set it to the actual model context limit, because this setting does not enlarge the model. When the endpoint is Ollama, MinAgent also reads the model's real runtime window from `POST /api/show` (`num_ctx`), so the effective window is the smaller of `OPENAI_CONTEXT_WINDOW` and that value even when the model name states nothing. A value above the model's real window lets the server truncate silently, and a large workspace inventory can then push the user's own question out of context. Too small a value is reported before the request instead, with the largest prompt components to trim. `WORKSPACE_LIST_LIMIT` defaults to `0`, which disables recursive inventory in the model context. `@` file autocomplete still searches a local index bounded to 10,000 entries and excludes common generated directories. The model can also call `list_directory` for a focused listing. A positive value includes up to that many entries per directory; `-1` includes all entries. Inventories stop at 10,000 entries or 128 KiB of text. `/init` builds a one-time inventory regardless of this setting. `TERMINAL_MODE` accepts lowercase `auto`, `ask`, or `off`, and defaults to `ask` when it is not set. `MCP_APPROVAL_MODE` takes the same three values for MCP tool calls and also defaults to `ask`: `ask` confirms each call, `off` refuses them all, and `auto` runs them without asking. In `auto` the call and its arguments are still printed before execution, so an unconfirmed call stays visible, and the system prompt tells the model that nobody will stop it. An MCP server executes with the user's own permissions, so `auto` means any configured server can act on this machine without a human in the loop.

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

When `OPENAI_SHOW_REASONING=on` and the endpoint supplies a reasoning delta, the reasoning is printed before the final response as muted gray text without a separate panel or background, wrapped to the terminal width so it stays aligned as a block. If the endpoint does not supply that field, MinAgent continues to show `Processing...` and the final response normally. The channel is also kept when nothing renders it: a model that finishes a turn having written only its reasoning is answered with that reasoning rather than reported as an empty response.

The model chooses when it needs workspace contents. With `WORKSPACE_LIST_LIMIT=0`, no recursive inventory is injected, while `@` file autocomplete remains available from a bounded local index. The model can call `list_directory` to inspect a specific directory's immediate entries. Otherwise, the inventory supplies paths but no file contents. MinAgent does not force an initial `read_file` call merely because files exist. When a request depends on project files, the model should call `read_file` before planning, diagnosing, or changing them. `list_directory` includes hidden entries, does not recurse, and returns at most 500 entries by default. Successful edits and writes are verified internally by MinAgent; the model does not need to read the same file back. After a failed edit, reread the file before retrying so the new edit is based on its current contents.

MinAgent verifies the persisted contents before a successful edit or write tool reports completion. A model response may request at most 16 tool calls; a turn may use at most `MAX_TOOL_ROUNDS` tool rounds (64 by default). Tool output stored in the conversation is compressed first (ANSI colour stripped, line endings normalised, blank-line and space runs collapsed, whole-JSON payloads minified) and, when it still exceeds `TOOL_PREVIEW_CHARS`, reduced to a head-and-tail preview. The omitted text is not discarded: it is stored zlib-compressed under `.minagent/tool-outputs/`, and the truncation note names an id the model passes to `recall_tool_output` to read any character range of the original back. Recall is exact, so a result is never permanently lost, and a single recall is capped so it cannot refill the window the archive just freed.

Keeping the result recoverable is what lets the inline preview stay small. Before, one result was allowed to occupy up to a quarter of the window - roughly 65,000 tokens on the default window - and whatever did not fit was gone. Now a single result costs at most `TOOL_PREVIEW_CHARS` (about 2,800 tokens by default) and the rest is one `recall_tool_output` call away. Compression earns its place here by making the off-window copy cheap to keep: an archive is capped at 64 MB and its oldest entries are pruned past that, a single result over 8 MB is not archived at all, and a session with no archive root falls back to the old truncation.

## Loading capabilities on demand

Sending every tool schema and every guidance section up front costs a fixed slice of the context window on every single request: measured on this project, 3.6k tokens - 44% of an 8k window - before the user types a word, whether or not the task needs a shell, a browser or a database.

So the prompt carries only an index: one line per capability, naming what it does, which tools it brings, and whether it is always loaded, loaded, or still on demand. The model asks for what the task needs with `load_capability`, and from the next request on the full schemas and the guidance for it ride along. Only two things are always loaded: the loader itself, and `recall_tool_output`, because a session that cannot read back what it archived is stuck. Everything else, reading files included, is one call away - and a call that arrives without it is answered with the name to load, so the cost of forgetting is one round trip rather than a broken task. A capability that the agent stops reaching for is dropped again after `CAPABILITY_IDLE_TURNS` idle turns, which is what stops a one-off detour from taxing every later request, and a new conversation starts from the always-loaded set again.

On this project that is 1,343 tokens of fixed prompt at startup instead of 2,259, and 2,243 for a task that has loaded reads, writes and the shell. Three things do the work. The loader does not repeat the catalogue: the index is the one copy of the names, so the schema stays small and a name that does not match comes back as an error that lists the real ones. The index names at most three tools per line and says how many more there are. And the core prompt states only the rules that hold whatever else is loaded - how to use the shell, the memory or the web is said once, by the capability that carries those tools, instead of twice.

Three things keep that safe. Guidance travels with the tools it explains: the shell instructions ride with the shell, a server's own instructions with that server's tools, because a bare schema list is not enough to drive either correctly. A tool that is not loaded is not callable, and the error names the capability to load rather than leaving a dead end: `web_search` reports that it belongs to `web`, which is not loaded. And a tool no capability claims could never be loaded at all, so a test asserts that every registered schema is claimed by exactly one group.

The index and the loaded guidance sit after the stable core and before the clock, so loading or unloading something changes the tail of the prompt rather than invalidating the cached prefix the provider can reuse.

## Governing the context while it fills

Loading on demand keeps the prompt small at rest, but a long task still grows: every tool result is real text, and a session that reads forty files carries them. So the context is measured before every request and, once it passes `CONTEXT_HIGH_WATERMARK` (75% of the window by default), given up in a fixed order - most tokens for the least loss first:

1. Old tool results become archive references. This is the only step that frees thousands of tokens rather than hundreds, and it is lossless: the full text is archived, and one `recall_tool_output` call brings back any of it.
2. Memory hints are dropped. They are a recall aid, and the same knowledge is one `recall` call away.
3. Capabilities the agent has not touched during the current turn are unloaded at once, instead of after the usual grace turns. Anything it has called since the turn started stays, so a task in flight never loses the tool it is using halfway through.
4. The capability index loses its summaries, and goes last on purpose: the names stay so anything can still be loaded, but stripping it early leaves the agent unable to tell what anything is for exactly when it is short on room.

How far down the cascade it goes depends on how far past the mark the context is: at 75% one step, at the top of the window all of them. A step the session has nothing to give is stepped over rather than counted, so a conversation with no old results does not stall on the first step. Nothing that could be lost is ever given up - the always-loaded tools, the core rules and the conversation itself are not on the list. Once the context drops under `CONTEXT_LOW_WATERMARK` (55%) everything that can be restored is. That gap is deliberate: without it a conversation hovering at the threshold would shed and restore on every turn, changing the shape of the prompt constantly and defeating the cache the ordering above protects.

`/context` shows what the governor is doing, and every shed is printed as it happens, so a prompt that changes shape is never a mystery. Compaction is not part of the cascade: the turn loop already runs it, and it wants the room these steps free.

### Is the meter telling the truth?

Every number on screen - the status bar, `/context`, the watermarks the governor uses - comes from a character-count heuristic, while the endpoint counts real tokens with its own tokenizer. A meter that reads 30% low is a governor that trims a third of the way too late, so the two are compared: MinAgent records what it predicted immediately before sending each request, and what the endpoint reported for that same request, and `/context` shows the drift and what it means for the thresholds. Getting the endpoint to report at all takes a request option, `stream_options.include_usage`, that a strict endpoint may reject; when it does, the field is dropped and the request is retried once, because the usage frame is worth asking for but not worth failing a turn over.

Against a local Ollama serving a 9B model the estimate runs about 3% under, so `CONTEXT_HIGH_WATERMARK=75%` really trims at roughly 73% of the window - close enough that the numbers on screen can be trusted, and worth checking on your own endpoint rather than assuming.

## Running reads together

When a response asks for several tools at once, the read-only ones run at the same time instead of queueing behind each other. A turn that read twelve files took eleven times longer than it needed to; now the whole batch costs about one read. Nothing about the context changes - the results still land in the transcript in the order the model asked for them - so this is purely wall clock, and it is the difference between a fifteen second turn and a two second one when the model wants to look at a handful of files.

The safety property is an allowlist, not a denylist. `read_file`, `list_directory`, `recall`, `recall_tool_output`, `load_skill`, `web_search` and `web_fetch` can overlap because none of them changes what another would see. Everything else stays strictly serial and in order: the writing tools, `run_terminal`, the memory writers, the skill and MCP authoring tools, and every MCP tool, whose side effects its own server decides and MinAgent cannot know. The batch also stops at the first call that is not on the list, so a read that follows a write never starts before that write is done. One read alone is not worth batching, and each read settles on its own, so one failing read cannot discard the results the others already produced.

`PARALLEL_TOOLS=off` restores the strictly sequential behaviour.

## Clearing old tool results

Before MinAgent resorts to summarising a conversation, it clears the old tool results out of it. Anthropic calls this the safest, lightest touch of compaction: once a tool result has been processed the model rarely needs its text again, and dropping it costs far less fidelity than summarising everything. `TOOL_RESULT_KEEP` (3 by default) recent results are kept verbatim; older ones are replaced with a stub that names an archive reference, so unlike Anthropic's server-side clearing nothing is lost - the model can bring any part of it back with `recall_tool_output`. A result that was already truncated keeps its existing reference instead of being archived twice, and nothing is cleared at all when there is nowhere to archive it. On a forty-turn editing session this frees about 91% of the transcript, which means the lossy summary step runs far less often.

Two notes on the trade-offs. Clearing invalidates the cached prefix where it happens, which is why it only runs when the conversation is already over the compaction threshold rather than after every turn. And editing earlier turns can invalidate reasoning blocks in later turns on reasoning models, so the cleared history is kept for recall rather than rewritten.

## Filtering tool output without asking the model

Alongside the lossless compression above, `compress_for_context` also drops things the model cannot use, and it does so without a second model call: HTML and XML wrappers, base64 blobs, and long runs of an identical line, which become a note saying how many times it repeated. That last one is a deliberate trade - a transcript repeating a line forty times carries no more signal than one saying it happened forty times. The guards matter, because an agent reads far more code than markup: tags are only stripped when at least one has a plausible tag name, so a comparison like `x < len(y) and z > 3` survives, and either a real share of the text is markup or the document has several tags, so a single `<b>` in a diff survives. Repeating a line, in test output, saves around 99%.

## Prompt caching and request replay

Providers only reuse a prompt prefix that is byte-identical to the one they already saw, and the system message is the first thing in the request, so a character that changes in it invalidates the whole reusable prefix. Two things keep that prefix stable. The host clock in the system prompt is truncated to the minute, because a second-resolution clock changed the system message on every single request and forced a cache miss each time; a minute of drift is irrelevant since the model can always run `date`. And the sections that legitimately change - the clock, `AGENTS.md`, the compacted summary, the workspace inventory, the memory hints - are placed after the stable ones instead of interleaved with them, so the reusable block covers as much of the prompt as it can.

On top of that, a request that is byte-identical to one already answered is replayed from a small in-memory LRU instead of being sent again. This earns nothing on the happy path, where every turn changes the prompt; it pays off on the paths that deliberately resend the same request, such as the retry after an empty or truncated response, a resubmitted prompt from the history ring, and a corrective nudge. The whole completion is replayed rather than just the text, because the caller branches on the `truncated` flag and on the usage figures. A cancelled, interrupted, or aborted response is never stored: the user is entitled to a fresh attempt. The cache holds 24 entries and is dropped on a new conversation or a model switch.

When the fixed prompt - system sections plus tool schemas - already uses at least 70% of the configured context window, startup prints a warning naming the components responsible and the settings that shrink them. If it reaches the compaction budget, the warning says the next request will be refused. Either way the cause is actionable before the first request, instead of an endpoint silently truncating the prompt.

When enabled, the inventory is refreshed before each model request. If the workspace root contains `AGENTS.md`, its content is reloaded before each request and included as project guidance up to 64 KiB, whether or not the inventory is enabled.

## Input, multiline text, and file attachments

Press `Ctrl+J` to insert a newline without sending the message. Multiline text pasted into the prompt keeps its line breaks and does not submit one request per line. Press Enter to send. Press ↑/↓ to recall previous inputs: the arrows first move between the buffer's own lines, then step back through the session's submitted inputs, and stepping past the newest entry restores what you were typing. While `@` or `/` autocomplete is open, ↑/↓ select a candidate instead.

While a response is streaming the prompt is not waiting for input, so keystrokes are ignored rather than collected into the next line; `Esc` still stops the request. A lone `Esc` is recognised after a 50 ms grace period, which is what tells it apart from the escape sequences that arrow keys and other special keys send.

Type `@` followed by a filename fragment to search workspace files. Use ↑/↓ to select a result and Enter to replace the fragment with its complete path in the current line; press Enter again to submit. Selecting a text file attaches an excerpt of up to 48 KiB. Selecting an image makes it available to the model, which loads it when the answer needs it (see below). Up to eight files and four images can be attached to one message; each file is limited to 10 MiB.

Image paths written directly in a message are detected for PNG, JPEG, GIF, and WebP files inside or outside the workspace. Outside images must be named explicitly; MinAgent does not list outside directories. The model endpoint must support image input.

By default an image is *noticed, not sent*: the message keeps the path and names the images it found, and the pixels only enter the context when the model loads the `images` capability and calls `view_image`. A single 1024×1024 screenshot is roughly 1000 tokens, a twenty-fifth of an 8k window, and most turns that mention an image never need the pixels. Loaded images are released again when the context fills: the encoding is dropped and the path stays, so the same call brings the picture back. `IMAGE_INPUT_MODE=eager` attaches images with the message instead, as previous versions did.

Set `NO_COLOR` to disable terminal colors.

## Commands

Type `/` to open command autocomplete. Use ↑/↓ to choose a command and Enter to complete it in the current line; press Enter again to run it. The available commands are:

- `/context`: show approximate token counts for system sections, available tool schemas, and conversation history, plus the latest endpoint-reported `prompt_tokens` when available.
- `/compact [instructions]`: summarize history older than the recent ~20,000-token window. Compaction cuts only at safe user or completed assistant-message boundaries, so a large completed tool round can be summarized as a unit.
- `/init [focus]`: inspect a one-time workspace inventory and selected project files, show which files were selected, and create or update the workspace root `AGENTS.md`. It reads up to 24 files, with excerpt and total-size limits.
- `/skills [reload | show <name> | delete <name>]`: list the registered skills, force a rescan, print one skill's instructions, or delete a skill that lives inside the workspace.
- `/memory [forget <id>]`: show how many memories exist, their success and reuse counts, and the strongest entries; `forget` deletes one entry.
- `/mejoras [now]`: show what past reflections concluded, or force a reflection on the current session.
- `/skill <what it should do>`: ask the model to draft a `SKILL.md` for that capability, register it immediately, and report the resulting name and path. An unfinished draft is reported; nothing is registered unless it validates.
- `/model [name]`: list the models the endpoint advertises through its OpenAI-compatible `/models` endpoint (Ollama and llama.cpp both expose it), marking the current one, or switch to `name` when given. While you type `/model `, ↑/↓ choose from a live picker and Enter completes the name. Switching persists `OPENAI_MODEL` in the project `.env` and warms the model with a one-token request so the first turn is not the load.
- `/doctor`: check the model, the context window, and the fixed prompt overhead.
- `/new`: clear the screen and start a new conversation. Everything the finished conversation owned is dropped, including the token tallies behind `/usage` and the replay cache, so an identical opening question is never answered from the previous run. The context bar still does not read empty afterwards: the system sections and the tool schemas are resent with every request, so the panel names that floor as `Fixed` and `/new` says so explicitly.
- `/exit`: close MinAgent.

Compaction also runs automatically as the usable context window fills. The usable window is the smaller of `OPENAI_CONTEXT_WINDOW` and any size stated by the model name (for example `...-8k`), so a model whose name states a smaller window compacts before the server truncates the prompt. The summary preserves file paths, decisions, unresolved work, user preferences, and verification state. It reduces conversation history; the system prompt, workspace guidance, inventory, and tool schemas remain. `/compact` reports both history and total context before and after, and `/context` shows the fixed prompt and tool-schema estimates.

## Workspace tools

The model can use these built-in tools; directory listings and file changes stay within the workspace root:

- `read_file`: read a specifically named UTF-8 text file inside or outside the workspace, or a supported image when image input is enabled. It cannot list directories. Text output is limited to 300 lines and 48 KiB. For a long line, use the returned `offset` and `column` to continue within that line. Part of the `files.read` capability.
- `view_image`: load an image's pixels so the model can see it. The path is the one from the conversation; the pixels arrive in the next request. Part of the `images` capability, which is loaded on demand like the rest.
- `download_file`: save a file from an http(s) URL. Every download lands in `salida/` and nowhere else: a path in the requested name is reduced to its last segment, a name that resolves outside the folder is refused, and a repeated download is numbered rather than overwriting. The cap is 64 MiB. Part of the `files.download` capability.
- `list_directory`: list immediate files and subdirectories, including hidden entries, without recursion. It defaults to the workspace root and 500 entries; pass a workspace-relative `path` or a larger `limit` when needed. Output is capped at 50 KiB and 10,000 entries; symbolic links are shown but never followed. Part of the `files.read` capability.
- `edit_file`: replace one exact, unique text block in an existing file. Part of the `files.write` capability.
- `write_file`: create or atomically replace one UTF-8 file, creating its missing parent directories. It writes a file, never a folder: a path ending in `/` is refused. Part of the `files.write` capability.
- `create_directory`: create a folder and any missing parent folders; an existing folder is reported as such. Part of the `files.write` capability.
- `delete_file`: delete one regular file. Part of the `files.delete` capability.
- `delete_directory`: recursively delete a regular subdirectory after validating its contents. Part of the `files.delete` capability.
- `run_terminal`: available only when `TERMINAL_MODE` is `auto` or `ask`, and part of the `terminal` capability. It runs in the workspace directory; `ask` requires approval for each command. It is the tool for system facts such as the current date and time, the environment, or installed tools.
- `recall`, `remember`, and `record_outcome`: available only when `MEMORY_ENABLED` is `on`, and part of the `memory` capability. `recall` searches the persistent memory before a task, `remember` saves a verified procedure or conclusion, and `record_outcome` reinforces or degrades a memory after it is reused.
- `web_search` and `web_fetch`: available only when `WEB_SEARCH_ENABLED` is `on`, and part of the `web` capability. `web_search` returns titles, URLs, and snippets; `web_fetch` reads one result page. Web content is untrusted data.

The prompt carries the host's local date and time, refreshed with every request, so a time question is answered from the real clock instead of a guess. When `TERMINAL_MODE` is not `off`, the prompt also states that `run_terminal` can read the rest of the system, and the shell instructions themselves arrive with the `terminal` capability. If a reply claims a capability is unavailable without calling any tool, MinAgent sends one corrective message naming the tools that are callable right now and the capabilities that are not loaded yet, and asks the model to use one, or to save a reusable skill with `write_skill` when something is genuinely missing, rather than ending the turn on "I have no access". A reply that only describes the next step ("voy a listar los correos") without calling a tool gets the same single nudge, so an announced plan is not mistaken for the work. A second refusal is returned as the answer, so the turn never loops over it.

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

### Deciding what is worth keeping

That automatic capture is a faithful log and a poor memory: most turns are a question that will not come up again, and a hint block built from all of them is mostly noise. Three layers decide what survives, cheapest first.

**In the turn.** The memory capability tells the model to call `remember` at the moment it notices something durable - a method that worked after several attempts failed, a constraint it discovered, a preference the user stated. This is free, because it already has the context, and it is the layer that gets the judgement right.

**Eureka.** Cheap signals mark a turn as a candidate and only then is the model asked whether it is worth keeping: a turn that failed two or more times and then worked, which is the shape of an actual discovery, or a turn of six or more steps, which is more method than a single call. A signal that fired on most turns would cost a model call on most turns, which is the same as having no signal, so the thresholds are deliberately conservative. A turn the model already remembered itself is never asked about again, and an answer that cannot be read as JSON keeps nothing: storing on a garbled answer is how a store fills up with guesses. Set `MEMORY_EUREKA=off` to skip it.

**A review.** Every `MEMORY_REFLECTION_INTERVAL` turns (10 by default) the auto-captured log is put in front of the model, which answers with the ids to keep and the ids to forget. A kept entry is reinforced and marked as judged, so it is never offered to a later cull; a forgotten one is deleted. An id the model invented is dropped rather than honoured, because forgetting the wrong memory is not recoverable. Set `MEMORY_REFLECTION_INTERVAL` higher for a longer backlog.

None of the three can fail a turn. The reply is already on screen by the time they run, so a reflection that fails costs a memory that was not written and never an error the user has to see.

### One memory said twice

Two memories that say the same thing are one memory said twice, and storing it twice fills the hint block with one answer in two voices and makes both look weaker than they are. The store merges on the content, not the title - the model invents its own title each time - and only for the knowledge kinds; the log of what was done is never merged, because two turns can look alike and still be two things that happened.

By default the comparison is lexical: whole words, technical identifiers counted once rather than as their parts, and a floor on how many distinctive words two memories must share before the ratios mean anything. That catches a paraphrase and leaves a near neighbour on the same subject alone, which is the whole job.

`MEMORY_EMBED_MODEL=nomic-embed-text` adds a second opinion from an embedding model on your Ollama, for the paraphrase that shares almost no words with what is stored. It is asked only about stored memories that share at least one distinctive word with the new one, and only about texts it has not embedded yet this session, so a save costs one fast request at worst. On a Spanish store the numbers are: a restatement of a stored memory scores 0.89 to 0.96, two different memories on the same subject 0.52, two on adjacent subjects 0.65, and the closest pair of genuinely different memories in the store 0.79 - so the bar sits at 0.80, above the worst real pair and below every measured duplicate.

The check can only ever *add* a merge. Below the bar the words decide, so a model that is not pulled, not running, or unsure costs one duplicate memory and nothing else; the store falls back to comparing words and stops asking. Two limits are worth knowing: it stays quiet on memories too short to tell apart (a one-line restatement scores the same 0.69 as a different fact on a different subject, and it declines to guess), and it does not fix the case the lexical rules miss either - the same fact in two languages, which an English-trained model reads as a near neighbour at 0.66.

### Improving itself

Knowing things is not the same as acting on them. `IMPROVEMENT_ENABLED=on` adds a reflection that asks what the session *implies*: a session where the same job was refused four times is not four facts, it is a hypothesis about a setting that is too tight. It runs when a session ends with `/new` - the one moment the whole arc is available, and the last one - and every `IMPROVEMENT_INTERVAL` reviews (10 by default) so a trend is caught before it costs a whole session.

What comes out are hypotheses, each with the evidence that supports it and the way to check whether it was right. They go to two places, because they are useful in two ways: `MEJORAS.md` at the project root, which the user reads and can edit, and the memory store, so a hypothesis resurfaces at the moment it is relevant. `/mejoras` shows the document, `/mejoras now` forces a pass.

A hypothesis may also ask for one of four settings to move, and **only** those move on their own - `MEMORY_REFLECTION_INTERVAL`, `COMPUTE_QUEUE_LIMIT`, `COMPUTE_JOB_TIMEOUT_SECONDS` and `COMPUTE_VOICE_TIMEOUT_SECONDS`. Each is a number with a minimum, a maximum and a 24-hour cooldown, written to `.env` and applied to the running session, and every change is listed in the document and in `.minagent/ajustes.json`. Set `IMPROVEMENT_AUTO=off` to keep the proposals and none of the moves.

Five rules decide whether a proposal becomes a change, and each exists because running it against the real 9B model showed what goes wrong without it:

- **Evidence is required.** A hypothesis with nothing behind it is written down and applied to nothing.
- **Only a named setting moves.** Anything else the model invents is dropped at the parse, before it can reach the file.
- **Bounded and nudged.** A value outside its range is refused, and so is a move of more than half the allowed span, which is a different setting wearing the same name.
- **One move per setting per day.** Without it, a hypothesis that is wrong in one direction gets corrected by the next one in the other, forever, with both looking supported.
- **An observation moves nothing.** Measured against this model: asked to reflect on a session where a render hit the time limit, it noticed that "renders take longer than the timeout" - an insight - and proposed a *shorter* timeout, the opposite of its own evidence. The bounds would have contained that damage, not prevented it.

`IMPROVEMENT_MODEL` runs the reflection on a different model of the same endpoint. It is empty by default, and that is a measurement rather than an omission: against the prompt a real session sends - 12 memories and 40 log lines, about 18000 characters - the 9B answered with hypotheses that could be parsed three times out of three, and `gemma4:12b-q3km`, which had looked better on a short curated prompt, answered in prose zero times out of four. A comparison on 900 characters of material does not transfer to 18000, which is why the number is worth having.

Naming a model is also not free on a card where the two do not fit together: the session model is unloaded for the pass and loaded back afterwards, whether the answer arrives or not, and an already-resident reflection model is left alone rather than cycled. An answer that is not in the requested shape is asked again on the session model before the pass is given up on, and a model that cannot answer at all is not an error - the session model answers the same prompt more weakly but answers it.

What the guards cannot do is check that a number moves in the right direction. Every value in range is survivable, which is what the bounds are for, but the reason a change is defensible is the evidence in the document, not the number in the file. Source code is deliberately not in the set: an agent that rewrites itself has no way to notice that it made things worse.

### Validating the change

The first version of this applied a change and never found out whether it was right, which is not a loop - it is a sequence of unverified edits and a growing pile of settings nobody chose. A change is now put on probation instead: the value moves, a baseline is taken, and at the end of a window of real turns it is judged and either kept or **reverted on its own**, with the verdict said out loud and `.env` left in the state the verdict implies.

What is measured is deliberately narrow, because a system that reported a capability number would be reporting one it cannot defend:

- **Failures per turn** - tool errors, jobs refused for VRAM, and jobs that failed. The primary number: it is the one that maps onto the user being blocked. Refusals and failures are counted by the orchestrator, where the decision is taken, rather than by reading an error message, so a rewording would not silently become a measurement change.
- **Tool-result tokens per turn** - the context the agent spends on the tools it chose to use. Prompt tokens are deliberately excluded: they mostly track how long the conversation is, which is not something a change can be judged on.

Both are **rates**. A session with fewer errors because it was shorter is not a better setting, and dividing by turns is what stops the loop from learning to end conversations early.

The verdict is "no harm, and some gain", with a five percent bar on either metric:

- Failures per turn getting worse by more than 5% reverts it. So does failures appearing where there were none.
- Tool tokens per turn getting worse by more than 5% reverts it too - **fewer failures bought with much more context is a trade, not an improvement**, and the one nobody asked for.
- A window under 8 turns is not judged at all. The trial stays open, because reverting on two clean turns is worse than leaving a change in place for a while.
- A change that neither helps nor hurts is reverted, since leaving it in spends that setting's cooldown on nothing.

One change is tried at a time. Two at once would make the verdicts unreadable: if the pair improved, nothing says which one did. The trial lives in `.minagent/prueba.json` and is written on every turn, because a measurement that restarted on every restart would let a change that was never judged sit there for ever. `/mejoras` shows what is being measured.

Measured on the 9B model this project runs: a reflection produced a *plannable* hypothesis - one that names an allowlisted setting and a number - in roughly **one session in three**. The other two produce an insight, or a hypothesis without evidence, and are recorded and applied to nothing. That is the honest state of it. The loop closes; how often it has anything to try is the model's own reliability, and no amount of prompt tuning measured here changed that by much.

## Web search

When `WEB_SEARCH_ENABLED=on` and `OLLAMA_API_KEY` is set, the model can reach the web through Ollama's hosted API. It is told to call `web_search` when it does not know how to do something, when a task has already failed three or more times, or when it needs current information, and `web_fetch` to read one result page in full. MinAgent also nudges the model toward `web_search` on its own after three tool errors in one turn, instead of letting it retry the same approach.

Web results are untrusted data: the tool descriptions and the prompt say so, and fetched content is bounded to 24 KiB per call. Requests go to `WEB_SEARCH_BASE_URL` (default `https://ollama.com/api`) with your `OLLAMA_API_KEY` and time out after `WEB_SEARCH_TIMEOUT_SECONDS` (default 120).

## Desktop control: keyboard and mouse

When `INPUT_ENABLED=on` the agent can drive the desktop: move the pointer, click, double-click, hold a button for a drag, turn the wheel, press key combinations, and type.

It goes through `ydotool`, writing to `/dev/uinput`, and **not** through `pyautogui`. That is not a preference. A KDE session here is Wayland, `pyautogui` speaks Xlib and sees an empty display, and `xdotool` can only reach XWayland clients, which on a modern desktop means not the native windows at all. `ydotool` events come from a kernel device, so the compositor cannot tell which client sent them and every window receives them. Install it once:

```bash
sudo pacman -S ydotool
systemctl --user enable --now ydotoold   # or the unit this project installs
```

Without `ydotoold` running, every command fails with `failed to connect socket`; that daemon is not optional.

Two things about this desktop are worth knowing. Nothing can ask a window what is under a coordinate — the pointer position is the only address there is — so the pattern is: take a screenshot, move the pointer, click, screenshot again. And the wheel is a relative count, so a scroll down sends a negative number behind a `--` separator, without which it is read as an unknown flag and scrolls nothing.

`INPUT_ENABLED` is off by default, because this is the one capability that moves the real pointer and presses real keys.

## Camera and microphone

When `SENSES_ENABLED=on` the agent can take a photo with the webcam (`capture_camera`) and record from the microphone (`record_microphone`). `CAMERA_DEVICE` selects the device (default `/dev/video0`) and `MICROPHONE_MAX_SECONDS` bounds a recording.

Both are read **only when a tool is called**, and both write an ordinary file into `salida/`, so a frame is something to look at with `view_image` and a recording is something `transcribe_audio` can read. A camera that runs while the session is idle would turn a local tool into a surveillance device; this one does not.

`record_microphone` returns the transcript as well as the path, because the recording was made to be read. That needs the local speech engine, so it also needs `COMPUTE_ENABLED=on`; without it the tool says so plainly and hands back the audio rather than pretending to have heard anything. An empty transcript is reported as silence, not as a failure.

Both devices can be held by another program, usually a video call, and on Wayland that is exclusive. `ffmpeg` says so rather than returning a blank frame. A frame that comes out nearly black is called out too: more exposure time does not fix a dark room, and handing a model a black JPEG just produces a confident description of darkness.

## Managing local models

When `OLLAMA_MODELS_ENABLED=on` the agent can list what the local Ollama server holds (`list_models`), read one in detail (`show_model`), create a derived model with its own system prompt (`create_model`), delete one (`delete_model`), and report what the machine can run (`hardware_report`).

`create_model` is Ollama's `/api/create`: a derived model points at a base that is already installed and carries a prompt, parameters, and template of its own. **No weights are copied.** That is what makes role-specific models cheap - a dozen of them cost kilobytes between them, share one base's memory, and only one is ever resident on the card at a time. It is the curl from the Ollama docs, with the base and the name validated first, because a model name becomes a directory and a derived model whose base is missing fails at the first request rather than at creation.

`hardware_report` answers "can this machine run it" from `nvidia-smi` and from the server's own view of what is resident. It deliberately prefers the measured VRAM footprint over the size on disk: on an 8 GB card this project's 6.14 GB model holds 5.11 GB resident, so a check built on the file size would refuse a model that is running. For a model that is not resident, the disk figure is reported as an upper bound rather than as an answer, because the card footprint is smaller and a definite "does not fit" would be wrong.

`delete_model` is permanent. The server has no undo.

`should_derive_model` answers the question that comes before `create_model`: is a derived model worth it, or is restating the prompt each turn cheaper? The deciding cost is the context window. A system prompt repeated in every request occupies the window on every turn - a 600-token role prompt is a real slice of an 8192-token window - while a derived model holds the same prompt in its own config and costs nothing per request. It does that arithmetic in tokens against the window and multiplies by how often the role comes up, and a long prompt used once does not clear the bar. Ask for a `needs` capability and it also checks the base can actually do the thing, because a prompt cannot add a capability: deriving "vision" from a text-only base would produce a model that looks right and fails identically every time, and a base that cannot even be read fails at the first request rather than at creation. When the base is unusable it says so and stops, instead of returning a shrug.

`push_model` publishes a model to a registry, and it is the only tool in this project that sends anything off the machine. It is off unless `OLLAMA_PUSH_ENABLED=on`, and a name with no host - `team/model` - goes to **public ollama.com**, which cannot be undone; `registry.example.com/team/model` is a private registry. The client refuses a public name outright, so the tool is not the way to publish by accident, and what it can do is push to a private registry the user already runs.

Which one you are about to do is stated in plain words before anything is sent: the destination, and at most how many GiB travel. A derived model shares its base's weights by digest, so if the base is already on the destination this is kilobytes rather than gigabytes - the 6.59 GB figure in `/api/tags` is what the model occupies locally, not what a push transfers.

Confirmation is heavier for the irreversible case on purpose. A private registry asks `y`; a public one requires typing the model's full name, because one extra line of typing is proportionate to publishing something nobody can take back. Outside the interactive terminal there is nobody to ask, so the call fails and says that nothing was sent - it never falls back to publishing quietly.

## Writing its own capabilities

When `SUBAGENTS_ENABLED=on` the agent can write a module for itself: a Python file defining `create_tools()`, the same shape the built-in capabilities use. It is loaded like any built-in, and its tools are called the same way.

**A durable module is confirmed in the terminal before anything is written**, and that is not ceremony. The file is Python the model wrote, and it runs on this machine with the user's permissions. This is the same reasoning that already gates authoring an MCP server. A git branch per module is where the change can be read before it matters - it is a review point, not a barrier, and nothing about a branch stops code from running. The module source is shown in the approval, and a module too long to show is refused rather than truncated, because approving the first 8000 characters of code that then runs is a guess, not an approval.

`ephemeral=true` writes nothing to the repository and needs no approval, which makes it the right choice for a helper needed this turn. Ephemeral modules are capped at `SUBAGENTS_MAX_EPHEMERAL` and dropped, so a store that only grows never ends up taxing the context window. `module_template` hands back a minimal working module so the first one is a copy rather than a guess.

## Project layout

- `src/minagent/app.py`: TUI, conversation loop, tool dispatch, and commands.
- `src/minagent/attachments.py` and `src/minagent/image.py`: file attachments and image handling.
- `src/minagent/input.py`: keyboard and mouse control through ydotool on Wayland.
- `src/minagent/senses.py`: on-demand camera capture and microphone recording.
- `src/minagent/ollama_models.py`: listing, inspecting, creating, deleting, and publishing local models.
- `src/minagent/subagents.py`: agent-written capability modules, durable or ephemeral.
- `src/minagent/images.py`: the `view_image` tool, which turns a path into pixels on request.
- `src/minagent/download.py`: the `download_file` tool and the `salida/` destination rule.
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

## Local media generation

Two standalone scripts, outside the package, for generating media on this machine. Neither is a
MinAgent tool: they are run from the terminal.

- `scripts/generar_imagen.py` — Qwen-Image-2.1 (GGUF, quantized) through diffusers, writing PNGs
  to `salida/`. The text encoder stays in RAM and only the embeddings reach the GPU, because in
  bf16 it does not fit alongside the denoiser on an 8 GB card.
- `scripts/generar_video.py` — Wan 2.1 T2V-1.3B through diffusers, writing MP4s to `salida/`. This
  is the light variant, chosen so it fits in 8 GB of VRAM: 480p, 81 frames, with the UMT5-XXL text
  encoder loaded in fp8 and `enable_model_cpu_offload()` moving the rest to system RAM. The 14B
  model does not fit on a consumer card and is not attempted.
- `scripts/compute/` — the backends behind the `compute` capability below, which the agent calls
  as tools instead of you running them by hand.

```bash
python scripts/generar_video.py "un dron sobrevolando una selva con niebla"
python scripts/generar_video.py "un gato" --pasos 20 --frames 49
python scripts/generar_video.py "una astronauta" --semilla 42     # reproducible
python scripts/generar_video.py --descargar-solo                  # weights only, no generation
```

The first run downloads about 29 GB from HuggingFace, of which 22.7 GB is the UMT5-XXL text encoder;
it is cached under `~/.cache/huggingface` and reused afterwards. Expect minutes, not seconds, per
clip on an 8 GB laptop GPU. `--frames` must be `4n+1` (81, 49, 33, 25): Wan denoises in blocks of
four, and any other frame count fails later inside the VAE with a shape error that hides the real
cause. `guidance_scale` of 1 ruins prompt adherence; the default is 5.0.

## GPU compute: voice, video, and music on one card

An RTX 4060 Laptop has 8188 MiB of VRAM. Two heavy models at once is not a slow configuration on
that card, it is impossible, so `COMPUTE_ENABLED=on` turns the constraint into a policy:

- **The voice engines stay resident.** `speak_text` (Kokoro) and `transcribe_audio` (whisper.cpp)
  hold under 2.5 GB together and answer immediately — including *while a video is rendering*,
  because they never queue behind the heavy lock.
- **Heavy jobs are serialized.** `generate_video` (LTX-Video) and `generate_music` (AudioLDM2) take
  the card one at a time, each in its own process with system RAM backing the layer offload.
- **Free VRAM is measured, not assumed.** A job that will not fit is refused before it starts,
  naming the process holding the memory. On this machine an Ollama server was once sitting on
  6390 of 8188 MiB, which is the difference between a render and an OOM. `compute_status` reports
  the same reading on demand.

```bash
COMPUTE_ENABLED=on
```

The tools are a `compute` capability, so they are loaded on demand like the rest and cost nothing
in a request that does not generate anything. The backends live in `scripts/compute/` and are
optional dependencies: a missing `diffusers` fails that one call with the install command, and
never stops the agent from starting. `.agents/skills/compute-gpu/SKILL.md` covers the install
steps and the frame-count rules.

**A separate environment for the backends.** The generation stack does not run on the agent's
interpreter. Kokoro's `misaki` requires Python below 3.13, and LTX-Video's SentencePiece tokenizer
needs `transformers` 4.x, so both break on the 3.14 the agent itself uses. `COMPUTE_PYTHON` points
the subprocess at a 3.12 environment; `.env.example` has the exact commands. This is the one piece
of setup that is not optional here, and it is the piece that most looks like it should not be
necessary.

**What runs where, measured on this machine** rather than assumed:

| job | backend | notes |
|-----|---------|-------|
| `speak_text` | Kokoro 82M | falls back to CPU when the card is full, so it never fails |
| `transcribe_audio` | whisper.cpp, `base` | CPU build here; the GPU flag is chosen from the binary, not the machine |
| `generate_video` | LTX-Video 2B | `sequential` offload; 17 frames in ~1 min, peaked at 696 MiB |
| `generate_music` | AudioLDM2 `music` | 8 s of audio in 25 s; 30 s is the ceiling per job |

`sequential` offload is the default because it is the one that fits while another model holds the
card. `group` is faster on an empty GPU and OOMs here. MusicGen, the obvious choice for music, is
no longer in diffusers at all, which is why the music backend is AudioLDM2.

**Queueing.** A render is minutes of GPU time, so asking for three videos one after another would
spend an hour of turn time with no feedback. `queue_job` accepts a job and returns an id
immediately, `compute_status` shows every waiting job's position, and `compute_result` reads one
back when it finishes. Heavy jobs go through one queue whichever way they were asked for, so a
queued job and a directly requested one still take the card one at a time. The queue is bounded
by `COMPUTE_QUEUE_LIMIT`, because an unbounded backlog of minutes-long jobs is not a feature.

A job that does not fit right now is **queued, not refused** — the thing usually holding the card
is another of these jobs, and refusing the second video of a two-video request would be a bug
wearing a policy's clothes. The VRAM decision is made at the front of the queue, where the
reading is the freshest there is.

**Freeing VRAM held by something else.** With `COMPUTE_UNLOAD_OLLAMA=on`, a job that still will
not fit runs `ollama stop` first, which expires the keep-alive on a resident model, and puts it
back when the job finishes - in a `finally`, so a failed or cancelled job restores it too. The
reload happens while the card is still held, so the next queued job cannot start against a model
that is halfway back, and the model is asked to stay resident for an hour, because the default
five minutes is shorter than a render. A reload that Ollama refuses is reported and not raised: a
warm model that could not be put back makes the next message slower, and must not throw away a
render that already worked.

The unload is off by default on purpose, even though the model comes straight back, because
evicting someone's warm model is their call. With it off, a refused job names the process holding
the memory and leaves the decision alone.

The same orchestrator is also exposed as an MCP stdio server in `.agents/mcp/compute/index.py`, so
any MCP client generates under the same VRAM budget rather than a second, divergent one.

## Tests

```bash
uv run pytest
```

The tests use `pytest` and `pytest-asyncio` and cover workspace files and their safety checks, attachments, context chunking, streaming responses, tool rounds, skill discovery and authoring, the injected host clock, the startup prompt-overhead warning, and no-access recovery, terminal approval and timeouts, the editor, terminal rendering, a local MCP HTTP server, a streaming endpoint that answers a clock question by running `date`, the persistent memory store, its tools, its prompt hints, and its automatic experience capture, and the Ollama-backed web search and fetch client against a mock HTTP transport.

## License and notice

MinAgent's own code is licensed under the [MIT License](LICENSE). See [NOTICE.md](NOTICE.md) for the Pi attribution. The project is a standalone implementation inspired by the Pi agent harness; it does not include Pi source files.
