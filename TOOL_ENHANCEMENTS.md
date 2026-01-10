# Tool System Enhancements - Web Search & Approval System

## Overview
This enhancement adds web search capabilities and a comprehensive tool approval system to the AI Lab Server.

## New Features

### 1. Web Search Tools

#### `web_search(query, max_results=5)`
- **Purpose**: Search the web for current information using DuckDuckGo
- **No API key required**: Uses `duckduckgo-search` library
- **Configurable results**: Default 5 results, max 10 (to minimize context usage)
- **Returns**: List of search results with title, snippet, and URL

**Example Usage**:
```python
# LLM will automatically call this when needing current information
"What are the latest Python 3.13 features?"
# Tool call: web_search(query="Python 3.13 new features", max_results=5)
```

#### `read_webpage(url, max_chars=3000)`
- **Purpose**: Fetch and extract clean text from webpages
- **Smart extraction**: Removes HTML, scripts, navigation, ads
- **Context-aware**: Default 3000 chars, configurable to preserve token budget
- **Returns**: Clean text content + metadata (title, length, truncation status)

**Example Usage**:
```python
# After web_search, LLM can read specific pages
"Read the Python 3.13 release notes"
# Tool call: read_webpage(url="https://docs.python.org/3.13/...", max_chars=3000)
```

### 2. Tool Approval System

#### Configuration (`server_config.py`)

```python
# Enable/disable approval system
TOOL_APPROVAL_MODE = True  # Set to False to auto-approve all tools

# Tools requiring user confirmation
TOOLS_REQUIRING_APPROVAL = [
    # File operations (destructive)
    "write_file",
    "delete_file",
    "move_file",
    "copy_file",
    "create_directory",

    # Web operations (external calls)
    "web_search",
    "read_webpage",
]
```

#### How It Works

1. **LLM requests tool**: Model wants to use `web_search` or `delete_file`
2. **Server pauses**: Detects tool requires approval, saves state
3. **Client prompts user**: Shows tool details in formatted table
4. **User decides**:
   - `y` = Approve this tool
   - `n` = Deny this tool
   - `a` = Approve all remaining
   - `d` = Deny all remaining
5. **Server continues**: Executes approved tools, skips denied ones
6. **Repeat if needed**: If more tools require approval, loop continues

#### User Experience

When a tool requires approval, the client displays:

```
⚠ Tool Approval Required
The AI wants to use 2 tool(s) that require your approval.

┌────────────────────────────────────┐
│ Tools Requesting Approval          │
├───────────┬────────────────────────┤
│ Tool      │ Arguments              │
├───────────┼────────────────────────┤
│ web_search│ {                      │
│           │   "query": "latest AI" │
│           │   "max_results": 5     │
│           │ }                      │
└───────────┴────────────────────────┘

Approve web_search(query=latest AI, max_results=5)? [y/n/a/d] (y):
```

#### API Endpoints

##### `POST /chat`
- Normal behavior, but now returns `ApprovalRequiredResponse` if tools need approval
- Response schema:
  ```json
  {
    "approval_required": true,
    "tools_pending": [
      {
        "tool_name": "web_search",
        "arguments": {"query": "...", "max_results": 5},
        "tool_call_id": "call_abc123"
      }
    ],
    "message": "The AI wants to use 1 tool(s) that require your approval."
  }
  ```

##### `POST /chat/approve`
- New endpoint for handling approval decisions
- Request schema:
  ```json
  {
    "decisions": [
      {"tool_call_id": "call_abc123", "approved": true},
      {"tool_call_id": "call_def456", "approved": false}
    ]
  }
  ```
- Returns: Standard `ChatResponse` with final result

## Security & Safety

### Protected Operations
The approval system helps prevent:
- **Accidental file deletion**: User confirms before `delete_file`
- **Unintended writes**: User approves before `write_file` or `create_directory`
- **Unexpected web access**: User knows when AI searches or reads external content
- **File moves/copies**: User confirms before reorganizing files

### Configurable Safety
- **Enable/disable globally**: Set `TOOL_APPROVAL_MODE = False` to auto-approve all
- **Customize tool list**: Add/remove tools from `TOOLS_REQUIRING_APPROVAL`
- **Per-tool granularity**: Some tools can require approval, others auto-execute

## Dependencies

New Python packages required:
```bash
pip install duckduckgo-search beautifulsoup4 lxml
```

- `duckduckgo-search`: Web search (no API key needed)
- `beautifulsoup4`: HTML parsing for webpage content extraction
- `lxml`: Fast XML/HTML parser (backend for BeautifulSoup)

## Architecture

### Server-Side Changes

1. **tools.py**: Added `web_search` and `read_webpage` tools
2. **server_config.py**: Added approval configuration
3. **ai_server.py**:
   - New `ServerState` fields for pending approvals
   - New response models (`ApprovalRequiredResponse`, `ToolApprovalRequest`, etc.)
   - Modified `/chat` endpoint to check for approvals
   - New `/chat/approve` endpoint to handle approval flow

### Client-Side Changes

1. **ai_client.py**:
   - Modified `send_message()` to detect approval requests
   - New `_handle_tool_approval()` method for user prompting
   - Recursive approval handling (supports chained approvals)

### Flow Diagram

```
User: "Search for latest AI news and read the top article"
  │
  ├─> Client sends to /chat
  │
  ├─> Server: LLM wants to call web_search
  │   ├─> Check TOOL_APPROVAL_MODE = True
  │   ├─> Check web_search in TOOLS_REQUIRING_APPROVAL
  │   └─> Return ApprovalRequiredResponse
  │
  ├─> Client: Display approval prompt
  │   └─> User approves (y)
  │
  ├─> Client sends to /chat/approve
  │
  ├─> Server: Execute web_search
  │   ├─> Get search results
  │   ├─> LLM processes results
  │   └─> LLM wants to call read_webpage
  │       ├─> Check approval again
  │       └─> Return ApprovalRequiredResponse
  │
  ├─> Client: Display second approval prompt
  │   └─> User approves (y)
  │
  ├─> Client sends to /chat/approve
  │
  ├─> Server: Execute read_webpage
  │   ├─> Fetch webpage content
  │   ├─> LLM synthesizes answer
  │   └─> Return final ChatResponse
  │
  └─> Client: Display assistant response
```

## Configuration Examples

### Strict Safety (Default)
```python
TOOL_APPROVAL_MODE = True
TOOLS_REQUIRING_APPROVAL = [
    "write_file", "delete_file", "move_file", "copy_file", "create_directory",
    "web_search", "read_webpage"
]
```

### Moderate Safety (Allow Web, Require File Approval)
```python
TOOL_APPROVAL_MODE = True
TOOLS_REQUIRING_APPROVAL = [
    "write_file", "delete_file", "move_file", "copy_file", "create_directory"
]
```

### Minimal Safety (Auto-approve All)
```python
TOOL_APPROVAL_MODE = False
TOOLS_REQUIRING_APPROVAL = []
```

### Custom Safety (Only Deletions)
```python
TOOL_APPROVAL_MODE = True
TOOLS_REQUIRING_APPROVAL = ["delete_file"]
```

## Testing

### Test Web Search
```bash
# Start server
python ai_server.py

# In client
python ai_client.py
> What are the latest developments in quantum computing?
# Should trigger web_search approval prompt
```

### Test Webpage Reading
```bash
> Search for Python 3.13 release notes and read them
# Should trigger two approvals: web_search, then read_webpage
```

### Test File Operations
```bash
> Create a file called test.txt with "Hello World"
# Should trigger write_file approval
```

### Test Approval Denial
```bash
> Delete all .log files in /tmp
# Approve search, then deny delete - observe graceful handling
```

## Limitations & Future Improvements

### Current Limitations
1. **Streaming not supported**: Approval flow only works with non-streaming `/chat`
2. **No batch approval UI**: Each tool requires individual y/n/a/d decision
3. **Web search rate limits**: DuckDuckGo may throttle heavy usage
4. **Context limits**: Large webpages truncated at `max_chars`

### Future Enhancements
- [ ] Add approval support for streaming endpoint
- [ ] Implement "trust this session" mode (approve all for this conversation)
- [ ] Add configurable web search backends (Bing, Google Custom Search)
- [ ] Implement intelligent webpage summarization (instead of truncation)
- [ ] Add retry logic for web requests
- [ ] Cache web search results to reduce duplicate requests
- [ ] Add URL allowlist/blocklist for `read_webpage`

## Backward Compatibility

All changes are **backward compatible**:
- Set `TOOL_APPROVAL_MODE = False` to restore original behavior
- Existing tools continue to work unchanged
- New tools are optional (only used if LLM calls them)
- API remains compatible (new endpoints are additions, not replacements)

## Performance Considerations

- **Web search**: ~1-3 seconds per query
- **Webpage reading**: ~1-5 seconds depending on page size
- **Approval overhead**: 0 seconds if auto-approved, user-dependent if manual
- **Context usage**: `max_results=5` and `max_chars=3000` defaults keep token usage reasonable

## Security Notes

### Web Search Safety
- Uses DuckDuckGo (no tracking, no API key exposure)
- Results are public information only
- No authentication or cookies sent

### Webpage Reading Safety
- 10-second timeout to prevent hanging
- User-Agent header set (polite crawling)
- Only fetches content, doesn't execute JavaScript
- Strips potentially dangerous elements (scripts, styles)

### Approval System Safety
- User must explicitly approve each dangerous tool
- Denied tools receive error message (not executed)
- State preserved across approval requests
- No automatic approval of chained operations

## Troubleshooting

### "duckduckgo-search library not installed"
```bash
pip install duckduckgo-search
```

### "beautifulsoup4 library not installed"
```bash
pip install beautifulsoup4 lxml
```

### Approval prompts not appearing
- Check `TOOL_APPROVAL_MODE = True` in `server_config.py`
- Verify tool is in `TOOLS_REQUIRING_APPROVAL` list
- Ensure using non-streaming client (`send_message()`, not `send_message_streaming()`)

### Web search returns no results
- DuckDuckGo may be rate-limiting - wait a few seconds
- Check network connectivity
- Try different search query phrasing

### Webpage reading fails
- Check URL is valid and accessible
- Some sites block automated requests
- Increase timeout if needed (modify `requests.get(timeout=10)`)

---

**Last Updated**: 2026-01-10
**Branch**: `claude/enhance-tools-web-search-7UXVO`
**Author**: Claude (AI Assistant)
