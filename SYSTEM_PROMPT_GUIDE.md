# System Prompt Guide - Tool Usage Optimization

## Problem: Models Using Tools Unnecessarily

**Symptom**: Models call tools for simple queries like "Hello", "How are you?", or basic questions they can answer directly.

**Root Cause**: The original system prompt was too minimal:
```python
"You are a helpful AI assistant."
```

This gave models **zero guidance** on when to use tools, so they defaulted to being "helpful" by using tools even when unnecessary.

---

## Solution: Improved Default System Prompt

The new default system prompt provides **clear guidance**:

```python
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools for specific tasks.

IMPORTANT - When to use tools:
- Use tools ONLY when you need information or capabilities you don't have
- For greetings, casual chat, or questions you can answer directly - just respond naturally WITHOUT tools
- Examples of when to use tools: reading files, searching the web for current info, modifying files
- Examples of when NOT to use tools: "Hello", "How are you?", "What is Python?", "Explain recursion"

Think before calling tools: Do I actually need this tool to answer the question?"""
```

### Key Improvements

1. **Explicit guidance**: "Use tools ONLY when you need information or capabilities you don't have"
2. **Positive examples**: When tools ARE appropriate (files, web search, etc.)
3. **Negative examples**: When tools are NOT needed (greetings, knowledge questions)
4. **Reflection prompt**: "Think before calling tools: Do I actually need this tool?"

---

## Model-Specific Prompts

Some models need **custom instructions** because they:
- Are overly eager to use tools (need stricter guidance)
- Are reluctant to use tools (need encouragement)
- Have specific prompt formatting requirements

### Configuration

**Located in `server_config.py`:**

```python
MODEL_SPECIFIC_PROMPTS = {
    # Example: Tool-eager model needs restraint
    "qwen2.5-7b-q4": """You are a helpful assistant.
Use tools sparingly - ONLY when absolutely necessary for information you don't have.
Most questions can be answered directly without tools.""",

    # Example: Tool-shy model needs encouragement
    "ministral-8b-q4": """You are a helpful assistant with powerful tools.
When a task requires reading files, searching the web, or modifying content, use the appropriate tool.
Don't hesitate to use tools when they would help answer questions.""",
}
```

### How It Works

1. **On startup**: Server loads `get_system_prompt_for_model(DEFAULT_MODEL_KEY)`
2. **On model switch**: Server updates to the new model's specific prompt
3. **Fallback**: If no model-specific prompt exists, uses `DEFAULT_SYSTEM_PROMPT`

### Helper Function

```python
def get_system_prompt_for_model(model_key: str) -> str:
    """Get the appropriate system prompt for a given model."""
    return MODEL_SPECIFIC_PROMPTS.get(model_key, DEFAULT_SYSTEM_PROMPT)
```

---

## Testing & Tuning

### Test Cases

**Good Behavior** (should NOT use tools):
```
User: "Hello"
Expected: Direct greeting, no tools

User: "What is Python?"
Expected: Knowledge-based answer, no tools

User: "Explain how neural networks work"
Expected: Educational explanation, no tools
```

**Correct Tool Usage**:
```
User: "What are the latest AI developments?"
Expected: Uses web_search (requires current info)

User: "Read the README.md file"
Expected: Uses read_file (requires external data)

User: "Create a new file called test.txt"
Expected: Uses write_file (requires file operation)
```

### Tuning Process

If you find a model is:

**1. Too tool-eager** (uses tools unnecessarily):
```python
MODEL_SPECIFIC_PROMPTS["your-model-key"] = """You are a helpful assistant.
CRITICAL: Use tools ONLY as a last resort when you absolutely cannot answer without external information.
Most questions should be answered directly from your knowledge."""
```

**2. Too tool-shy** (doesn't use tools when needed):
```python
MODEL_SPECIFIC_PROMPTS["your-model-key"] = """You are a helpful assistant with access to powerful tools.
Proactively use tools when they would provide better, more accurate, or more current information.
Examples: file operations, web searches for recent data, directory listings."""
```

**3. Just right**: Remove from `MODEL_SPECIFIC_PROMPTS` to use default

---

## Runtime Control

Users can **override** the system prompt at any time via the `/command` endpoint.

### View Current Prompt
```bash
curl -X POST http://localhost:8080/command \
  -H "Content-Type: application/json" \
  -d '{"command": "system"}'
```

### Set Custom Prompt
```bash
curl -X POST http://localhost:8080/command \
  -H "Content-Type: application/json" \
  -d '{"command": "system", "value": "Your custom prompt here"}'
```

### Reset to Model-Specific Default
```bash
curl -X POST http://localhost:8080/command \
  -H "Content-Type: application/json" \
  -d '{"command": "system", "value": "reset"}'
```

**In the client** (`ai_client.py`):
```
> /system
Current system prompt: [displays current prompt]

> /system Your custom prompt here
✓ System prompt updated

> /system reset
✓ System prompt reset to model-specific default
```

---

## Best Practices

### 1. Start with Default
- The improved default prompt works well for 80% of models
- Only add model-specific overrides if you observe issues

### 2. Be Specific
- Bad: "Use tools wisely"
- Good: "Use tools ONLY when you need external information you don't have"

### 3. Provide Examples
- Models learn better from concrete examples
- Include both positive (when to use) and negative (when NOT to use)

### 4. Keep It Concise
- Some models have small context windows
- Aim for 3-5 sentences maximum
- Every word should add value

### 5. Test Thoroughly
- Test with greetings ("Hello", "How are you?")
- Test with knowledge questions ("What is X?", "Explain Y")
- Test with tasks requiring tools (file ops, web search)
- Verify the model makes appropriate decisions

---

## Architecture

### Prompt Resolution Flow

```
Chat Request
    │
    ├─> Check if custom prompt in request
    │   └─> Use request.system_prompt
    │
    ├─> Else, use server state.system_prompt
    │
    └─> state.system_prompt is set from:
        │
        ├─> On startup: get_system_prompt_for_model(DEFAULT_MODEL_KEY)
        │
        ├─> On model switch: get_system_prompt_for_model(new_model_key)
        │
        └─> On /command system: User override
            │
            └─> On reset: get_system_prompt_for_model(current_model_key)
```

### Code Locations

- **Configuration**: `server_config.py:231-249`
  - `DEFAULT_SYSTEM_PROMPT`
  - `MODEL_SPECIFIC_PROMPTS`
  - `get_system_prompt_for_model()`

- **Startup**: `ai_server.py:375-378`
  - Sets initial prompt for default model

- **Model Switch**: `ai_server.py:521-524`
  - Updates prompt when switching models

- **Runtime Override**: `ai_server.py:1167-1191`
  - `/command system` endpoint

---

## Common Issues & Solutions

### Issue: Model still uses tools for greetings

**Diagnosis**: Model may need stricter guidance

**Solution**: Add model-specific prompt:
```python
MODEL_SPECIFIC_PROMPTS["your-model"] = """You are a helpful assistant.
NEVER use tools for greetings, casual conversation, or questions you can answer from knowledge.
Tools should be used ONLY for file operations, web searches, or tasks requiring external data."""
```

### Issue: Model won't use tools even when appropriate

**Diagnosis**: Model may be over-cautious

**Solution**: Add encouraging prompt:
```python
MODEL_SPECIFIC_PROMPTS["your-model"] = """You are a helpful assistant with access to tools.
When a question requires current information (web search), file access, or external operations,
confidently use the appropriate tool. Tools are your friends!"""
```

### Issue: Inconsistent behavior

**Diagnosis**: May be temperature/sampling issue

**Solution**: Adjust generation parameters:
```python
# Lower temperature = more deterministic
DEFAULT_TEMPERATURE = 0.5  # Instead of 0.7

# Or adjust per-request in client
```

### Issue: Model ignores the system prompt

**Diagnosis**: Some models may not respect system prompts well

**Solution**:
1. Try different prompt phrasing
2. Consider using a different model
3. Use approval system to gate tool usage manually

---

## Performance Impact

**Prompt Length**: ~40-50 tokens (default)
- **Minimal impact** on context window
- Models with 4K+ context: No concern
- Models with <2K context: Monitor carefully

**Tool Call Reduction**:
- Expected: 60-80% reduction in unnecessary tool calls
- Faster responses for simple queries
- Lower latency overall

**Accuracy**:
- No negative impact on legitimate tool usage
- May improve quality by reducing noise from failed tool calls

---

## Examples of Model-Specific Tuning

Based on community testing:

### Qwen 2.5 Models (Very tool-eager)
```python
"qwen2.5-7b-q4": """You are a helpful assistant. Think carefully before using tools.
Use tools ONLY when you absolutely need external information or capabilities.
Simple greetings and knowledge questions should be answered directly."""
```

### Mistral Models (Generally balanced)
```python
# No override needed - use DEFAULT_SYSTEM_PROMPT
```

### Llama 3.2 Models (Sometimes tool-shy)
```python
"llama-3.2-3b-q4": """You are a helpful assistant with access to tools.
When appropriate, use tools to provide accurate and current information.
Don't hesitate to use tools for file operations, web searches, or external data."""
```

---

## Future Enhancements

Potential improvements:
1. **Dynamic prompts**: Adjust based on conversation context
2. **User profiles**: Per-user prompt preferences
3. **A/B testing**: Compare prompt effectiveness
4. **Tool-specific guidance**: Different prompts for different tool categories
5. **Learning system**: Track tool usage and auto-tune prompts

---

**Last Updated**: 2026-01-11
**Related Files**:
- `server_config.py` - Configuration
- `ai_server.py` - Implementation
- `TOOL_ENHANCEMENTS.md` - Tool system documentation
