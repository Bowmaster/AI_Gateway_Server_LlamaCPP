# Web Search Troubleshooting Guide

## Quick Diagnosis

If web search isn't working, run the diagnostic script:

```bash
python test_web_search.py
```

This will test:
1. Library installation
2. Network connectivity
3. Basic DuckDuckGo search
4. Integration with tools.py

## Common Issues & Solutions

### 1. Library Not Installed
**Symptom:** Import error messages
**Solution:**
```bash
pip install duckduckgo-search
# Or upgrade if already installed
pip install --upgrade duckduckgo-search
```

### 2. No Results Returned
**Symptoms:**
- Server logs show "No results found"
- Tool returns empty results array

**Possible Causes:**

**A. Rate Limiting**
- DuckDuckGo may be rate-limiting your IP
- **Solution:** Wait 5-10 minutes, then try again
- **Prevention:** Don't run searches too frequently

**B. Network/Firewall Issues**
- Corporate firewall blocking DuckDuckGo
- Proxy configuration needed
- **Test:** Run `python test_web_search.py` to check connectivity
- **Solution:** Configure proxy or use different network

**C. Query Too Specific/Obscure**
- Search query is too narrow
- **Solution:** Try broader search terms
- Example: Instead of "Python 3.13.2 bugfix", try "Python 3.13"

### 3. Timeout Errors
**Symptom:** "Search API error: TimeoutError"
**Causes:**
- Slow internet connection
- DuckDuckGo temporarily slow

**Solutions:**
```python
# Increase timeout in tools.py line 1387:
ddgs = DDGS(timeout=30)  # Increase from 20 to 30 seconds
```

### 4. Library API Changes
**Symptom:** Attribute errors, unexpected types
**Solution:**
```bash
# Check version
pip show duckduckgo-search

# If outdated (< 4.0):
pip install --upgrade duckduckgo-search

# If too new and broken, pin to known working version:
pip install duckduckgo-search==4.1.1
```

### 5. Results But Not Reaching LLM
**Symptoms:**
- test_web_search.py works
- Server logs show "Found X results"
- But LLM doesn't see them

**Check:**

1. **Approval System**
   - Is `TOOL_APPROVAL_MODE = True`?
   - Are you approving the tool when prompted?
   - Check client output for approval prompts

2. **Tool Execution**
   - Check server logs for "Tool call APPROVED"
   - Check for "Tool result:" in logs

3. **Sanitization Issues**
   - Results might be over-sanitized
   - Temporarily disable: `WEB_CONTENT_SANITIZATION = False`
   - If this fixes it, tune `WEB_CONTENT_AGGRESSIVE_SANITIZATION = False`

## Enable Detailed Logging

For debugging, enable DEBUG level logging:

**In `server_config.py`:**
```python
LOG_LEVEL = "DEBUG"  # Change from "INFO"
```

Restart server, then check logs for:
```
Web search initiated: query='...', max_results=5
Search results type: <class 'list'>
Result 0: {'title': '...', 'href': '...', 'body': '...'}
Found 5 results for query: '...'
```

## Testing Workflow

### Step 1: Test Library Independently
```bash
python test_web_search.py "Python programming"
```

Expected output:
```
✓ duckduckgo-search imported successfully
✓ Successfully connected to DuckDuckGo
✓ Successfully retrieved 5 results
✓ tools.py web_search working correctly
```

### Step 2: Test in Server
```bash
# Start server
python ai_server.py

# In another terminal, start client
python ai_client.py
```

Try query:
```
> What are the latest developments in quantum computing?
```

**Expected flow:**
1. Server logs: "Web search initiated: query='latest developments quantum computing'"
2. Client shows: "⚠ Tool Approval Required"
3. You approve with 'y'
4. Server logs: "Tool call APPROVED: web_search(...)"
5. Server logs: "Found X results for query: '...'"
6. Server logs: "Tool result: {'success': True, 'count': X, ...}"
7. LLM synthesizes answer from results

### Step 3: Check Server Logs

Look for these patterns:

**✓ Good:**
```
INFO - Web search initiated: query='quantum computing', max_results=5
DEBUG - Search results type: <class 'list'>
DEBUG - Result 0: {'title': '...', ...}
INFO - Found 5 results for query: 'quantum computing'
INFO - Tool call APPROVED: web_search(...)
INFO - Tool result: {'success': True, 'count': 5, ...}
```

**✗ Bad (Rate Limited):**
```
INFO - Web search initiated: query='...', max_results=5
ERROR - DDGS search error: RatelimitException: ...
```

**✗ Bad (No Results):**
```
INFO - Web search initiated: query='...', max_results=5
WARNING - No results found for query: '...'
INFO - Tool result: {'success': False, 'error': 'No results found...'}
```

## Workarounds

### If DuckDuckGo is consistently failing:

1. **Wait and retry** (rate limits usually clear in 5-10 min)

2. **Use different network** (VPN, mobile hotspot, etc.)

3. **Manual web lookup + read_webpage**:
   ```
   > Please read the content from https://en.wikipedia.org/wiki/Quantum_computing
   ```

4. **Disable web tools temporarily**:
   ```python
   # In server_config.py
   TOOLS_REQUIRING_APPROVAL = [
       # Comment out web tools
       # "web_search",
       # "read_webpage",
   ]
   ```

## Advanced Debugging

### Custom Test Query
```bash
python test_web_search.py "your specific query here"
```

### Check DuckDuckGo Directly
```python
from duckduckgo_search import DDGS

ddgs = DDGS(timeout=20)
results = ddgs.text("test query", max_results=3)

for r in results:
    print(r)
```

### Network Trace
```bash
# Install mitmproxy for HTTPS inspection
pip install mitmproxy

# Run with proxy
export HTTPS_PROXY=http://localhost:8080
python test_web_search.py
```

## Known DuckDuckGo Limitations

1. **Rate Limits**
   - ~30-50 searches per IP per hour
   - More aggressive if rapid-fire requests
   - Solution: Space out searches

2. **Regional Blocking**
   - Some regions/networks block DuckDuckGo
   - Solution: Use VPN or different provider

3. **API Instability**
   - DuckDuckGo doesn't provide official API
   - The library uses web scraping
   - Can break with site changes
   - Solution: Keep library updated

4. **No API Key**
   - Benefit: Free, no registration
   - Drawback: No guaranteed SLA

## Alternative Solutions

If DuckDuckGo continues to fail:

### Option 1: Use read_webpage with known URLs
```
> Go to Google and search for X, then tell me the URL
> [You paste URL]
> Read the content from [URL]
```

### Option 2: Add Bing/Google Custom Search
See TOOL_ENHANCEMENTS.md for future enhancement ideas

### Option 3: Use model's knowledge
```
> Based on your training data, what do you know about X?
> Note: This may be outdated if after your knowledge cutoff
```

## Success Indicators

Web search is working when:

1. ✓ `test_web_search.py` shows all tests passing
2. ✓ Server logs show "Found X results" (X > 0)
3. ✓ Tool result has `"success": True`
4. ✓ LLM mentions information from search results
5. ✓ Response includes URLs from results

## Still Not Working?

1. **Check library version:**
   ```bash
   pip show duckduckgo-search
   ```

2. **Try different query:**
   - Generic: "Python programming"
   - Specific: "What happened today"

3. **Check network:**
   ```bash
   curl https://duckduckgo.com
   ```

4. **Review full logs:**
   ```bash
   python ai_server.py 2>&1 | tee server.log
   ```

5. **Report issue:**
   - Include output from `test_web_search.py`
   - Include relevant server logs
   - Include duckduckgo-search version

---

**Last Updated:** 2026-01-12
**Related Files:**
- `tools.py` - Web search implementation
- `test_web_search.py` - Diagnostic script
- `server_config.py` - Logging configuration
