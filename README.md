# AI Lab (llama.cpp Edition)

FastAPI-based server that wraps llama-server for model management, conversation history, and tool calling support.

## Architecture

```
ai_server.py (FastAPI, port 8080)
    ↓
llama_manager.py (Process management)
    ↓
llama-server.exe (Native inference, port 8081)
```

## Quick Start

### 1. Setup

Make sure you have:
- Python 3.9+
- llama-server.exe in the project directory
- At least one GGUF model in the `models/` directory

Install dependencies:
```powershell
pip install fastapi uvicorn requests psutil
```

### 2. Configuration

Edit `server_config.py` to configure:
- **GPU/CPU mode**: Set `n_gpu_layers` (-1=GPU, 0=CPU, N=hybrid)
- **Model paths**: Update MODELS registry with your GGUF files
- **Default model**: Set `DEFAULT_MODEL_KEY`

### 3. Start Server

```powershell
python ai_server.py
```

The server will:
1. Validate configuration
2. Start llama-server as a subprocess
3. Load the default model
4. Listen on port 8080 for client connections

### 4. Test

```powershell
# In another terminal
python test_server.py
```

This runs 5 tests:
1. Health check
2. Models list
3. Simple chat (no tools)
4. Chat with tools
5. Custom system prompt

## Server Endpoints

### Health Check
```
GET /health
```

### List Models
```
GET /models
```

### Switch Models
```
POST /model/switch
{
  "model_key": "qwen2.5-7b-q4"
}
```

### Chat
```
POST /chat
{
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 1024,
  "enable_tools": true
}
```

### Commands
```
POST /command
{
  "command": "system",
  "value": "You are a helpful assistant"
}
```

Available commands:
- `system` - Get/set system prompt
- `layers` - Get GPU layer info
- `mem` - Memory usage
- `tools` - Enable/disable tool calling

## Configuration Modes

### GPU Mode (Default)
```python
"n_gpu_layers": -1  # All layers on GPU
```

### CPU Mode
```python
"n_gpu_layers": 0  # All layers on CPU
```

### Hybrid Mode
```python
"n_gpu_layers": 20  # 20 layers on GPU, rest on CPU
```

## Files

- `ai_server.py` - Main FastAPI server
- `llama_manager.py` - Subprocess management for llama-server
- `server_config.py` - Configuration and model registry
- `tools.py` - Native Python tools for function calling
- `test_server.py` - Test suite

## Features

✅ Model switching without restart (via /v1/models/load)  
✅ Conversation history management (stateful)  
✅ Tool/function calling support  
✅ OpenAI-compatible API  
✅ GPU/CPU/Hybrid modes  
✅ Process monitoring and health checks  

## Adding Models

1. Download GGUF model to `models/` directory
2. Add entry to `MODELS` registry in `server_config.py`:

```python
"my-model-q4": {
    "name": "My Model Q4",
    "filename": "my-model-q4.gguf",
    "description": "Description here",
    "context_length": 4096,
    "vram_estimate": "~5GB",
    "recommended": False,
    "download_url": "https://...",
    "usage": "Use case description"
}
```

3. Switch to it via `/model/switch` endpoint

## Troubleshooting

**Server won't start:**
- Check `llama-server.exe` exists
- Check default model exists in `models/`
- Check port 8080 and 8081 are available

**Model won't load:**
- Verify GGUF file exists
- Check VRAM availability (use hybrid/CPU mode if needed)
- Check llama-server logs

**Tool calling not working:**
- Ensure model supports function calling (Qwen2.5, Llama 3.2, etc.)
- Set `enable_tools: true` in chat request
- Check tool definitions in `tools.py`

## Next Steps

- Create client (`ai_client.py`) for CLI interaction
- Add more tools (`tools.py`)
- Implement MCP support
- Add model auto-downloading
- Build coding assistant features
