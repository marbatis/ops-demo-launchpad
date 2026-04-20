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
- The raw upstream `colab-mcp` design depends on dynamic tool refresh. The stable `colab` proxy avoids that and is the default for normal work.
