# Codex + Colab Workflow

This is the stable workflow for driving a Google Colab notebook from Codex.

## Server names

- `colab`: default stable server backed by the static compatibility proxy
- `colab_raw`: raw Google `colab-mcp` wrapper kept only for debugging

Use `colab` unless you are explicitly debugging MCP behavior.

## Fresh chat prompt

Paste this into a fresh Codex chat:

```text
Use the MCP server named colab.
First call get_colab_connection_status.
If connected, inspect the first 5 notebook cells.
If not connected, call open_colab_browser_connection exactly once, wait for me to click Connect in Colab, then call get_colab_connection_status again and inspect the first 5 notebook cells.
```

## Connection rules

1. Open one blank Colab notebook in Chrome.
2. Start a fresh Codex chat.
3. Let Codex call `open_colab_browser_connection`.
4. In Colab, click `Connect` when the local MCP dialog appears.
5. Continue with `get_cells`, `add_code_cell`, `update_cell`, `run_code_cell`, and `delete_cell`.

## Notebook durability rules

1. Do not keep important work in `scratchpad` or `empty.ipynb`.
2. As soon as the notebook opens, click `Copy to Drive` and give it a real name.
3. Treat the notebook document and the runtime filesystem as separate things:
   - `Copy to Drive` saves the notebook file.
   - It does not save files written under `/content`.
4. Treat `/content` as disposable VM storage.
5. After any expensive step, assume the runtime can disappear unless artifacts have been copied out explicitly.

## Artifact backup rules

Export important files immediately after they are created. For Nemotron-style workflows, this means:

- `submission.zip`
- the adapter directory if you may need to repackage or inspect it later
- a small manifest or summary JSON with paths, metrics, and timestamp

Back them up to at least one durable location outside `/content`:

- local download to your machine
- Google Drive
- GitHub for code, scripts, docs, and packaging helpers only

Do not rely on GitHub for large model weights or Colab VM state.

## Durability checklist

Before long runs:

- Notebook is Drive-backed and has a real name
- Secrets are configured and notebook access is enabled
- Output directories are known in advance
- You know which artifacts must be exported if the run succeeds

After every meaningful milestone:

- If `submission.zip` exists, download it immediately
- If an adapter directory exists, copy it out if it may be needed again
- Save a manifest JSON with the exact output paths and metrics
- Push workflow code and packaging helpers to GitHub

Use this question after any step that took more than a few minutes:

- If the runtime dies right now, what did I lose?

If the answer includes `submission.zip`, adapter weights, or a hard-won result, export it before continuing.

## Notebook tools available through the stable proxy

- `open_colab_browser_connection`
- `get_colab_connection_status`
- `list_colab_notebook_tools`
- `get_cells`
- `add_code_cell`
- `add_text_cell`
- `update_cell`
- `move_cell`
- `run_code_cell`
- `delete_cell`

## Known caveats

- Do not start from the Colab-side token dialog first. Codex must generate the connection flow.
- Keep only one actively connected Colab notebook when possible.
- If a chat gets confused, use a fresh Codex chat rather than fighting stale tool state.
- A named Drive-backed notebook helps with reconnecting to the right tab, but it does not protect `/content` artifacts by itself.
- The raw upstream `colab-mcp` design depends on dynamic tool refresh. The stable `colab` proxy avoids that and is the default for normal work.
