# Official MCP server

Install the optional MCP v2 dependency:

```bash
pip install "memoripy[mcp]"
```

Run a local stdio server:

```bash
memoripy mcp ./.memoripy
```

The server exposes:

- `memoripy_capture`
- `memoripy_recall`
- `memoripy_explain`
- `memoripy_correct`
- `memoripy_forget`
- `memoripy_audit`
- `memoripy_list`

Network transports are refused unless a token file is supplied:

```bash
memoripy mcp ./.memoripy \
  --transport streamable-http \
  --host 127.0.0.1 \
  --port 8000 \
  --token-file ./mcp-tokens.json
```

Example token file:

```json
{
  "tokens": {
    "replace-with-a-long-random-token": {
      "client_id": "claude-code",
      "subject": "khazar",
      "scopes": ["memoripy:read", "memoripy:write"]
    }
  }
}
```

The network server uses the MCP Python SDK's protected-resource and bearer-token validation surfaces. Stdio is intentionally allowed without a bearer token because process launch is the local trust boundary.

Use `--read-only` when the connected agent should be able to recall and explain memory but not create, correct, or forget records.
