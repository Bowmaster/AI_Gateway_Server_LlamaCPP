# Enhanced AI Lab Client

An improved command-line client for the AI Lab server with advanced input handling, autocomplete, and better copy/paste support.

## Features

- **Multiline Input**: Press Alt+Enter (or Escape then Enter, or Ctrl+J) to add new lines
- **Command Autocomplete**: Press Tab to autocomplete slash commands
- **History Search**: Press Ctrl+R to search through command history
- **Better Copy Support**: `/raw` toggle for plain text output, `/copy` for clipboard
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Configurable Server URL**: Via environment variable or config file

## Installation

```bash
# Navigate to the EnhancedClient folder
cd EnhancedClient

# Install dependencies
pip install -r requirements.txt

# Optional: Install clipboard support
pip install pyperclip
```

## Configuration

Server URL can be configured in three ways (priority order):

### 1. Environment Variable (Highest Priority)

```bash
# Windows (PowerShell)
$env:AI_SERVER_URL = "http://ailab.local:8080"
python ai_client_enhanced.py

# Windows (CMD)
set AI_SERVER_URL=http://ailab.local:8080
python ai_client_enhanced.py

# Linux/macOS
export AI_SERVER_URL="http://ailab.local:8080"
python ai_client_enhanced.py
```

### 2. Config File

Copy `config.example.yaml` to `config.yaml` and modify:

```yaml
server_url: "http://ailab.local:8080"
```

### 3. Default

If no configuration is provided, defaults to `http://localhost:8080`.

## Usage

```bash
python ai_client_enhanced.py
```

### Input Controls

| Key Combination | Action |
|----------------|--------|
| Enter | Submit message |
| Alt+Enter | Insert new line |
| Escape, Enter | Insert new line (alternative) |
| Ctrl+J | Insert new line (alternative) |
| Tab | Autocomplete command |
| Ctrl+R | Search command history |
| Ctrl+C | Interrupt current operation |

### Commands

#### Client Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help message |
| `/exit`, `/quit` | Exit client |
| `/reset` | Clear conversation history |
| `/model`, `/models` | Show and select models |
| `/models info` | Detailed model information |
| `/export <path>` | Export conversation to text file |
| `/export-json <path>` | Export to JSON |
| `/export-ft <path>` | Export for fine-tuning (JSONL) |
| `/status` | Show server status |
| `/hardware` | Show hardware configuration |
| `/stream [on\|off]` | Toggle streaming mode |
| `/raw` | Toggle raw output (for easy copy/paste) |
| `/copy` | Copy last response to clipboard |

#### Working Directory Commands

| Command | Description |
|---------|-------------|
| `/cd <path>` | Set working directory for file operations |
| `/cd` | Show current working directory |
| `/pwd` | Show current working directory |
| `/ls` | List files in working directory |

#### Server Commands

| Command | Description |
|---------|-------------|
| `/system` | Show current system prompt |
| `/system <prompt>` | Set system prompt |
| `/system reset` | Reset to default |
| `/layers` | Show GPU layer count |
| `/layers <N>` | Set GPU layers (-1=all, 0=CPU) |
| `/tools [on\|off]` | Enable/disable tool calling |
| `/mem` | Show memory usage |
| `/stop-server` | Shutdown server |

## Copy/Paste Workflow

For copying AI responses:

1. **Toggle raw mode**: Type `/raw` to switch to plain text output (no panels)
2. **Copy manually**: Select text in terminal and copy
3. **Or use /copy**: Type `/copy` to copy last response to clipboard (requires `pyperclip`)

## Working Directory

Set a working directory to tell the AI where to perform file operations:

```
/cd C:\Projects\MyApp
```

Once set, the AI will use this directory as the base for all file operations. This is useful when:
- Creating multi-file projects
- Reading/writing files in a specific location
- Working on an existing codebase

The working directory is included in the system prompt sent to the model, so it knows where to perform file operations.

```
You: /cd ~/projects/my-app
Working directory set to: /home/user/projects/my-app
Contains: 3 folders, 5 files

You: Create a Python Flask app with app.py, routes.py, and templates/index.html
```

## Examples

### Multiline Input

```
You: Write a Python function that:
... - Takes a list of numbers
... - Returns the sum of even numbers

(Press Alt+Enter or Ctrl+J after each line, then Enter to submit)
```

### Command Autocomplete

```
You: /str<Tab>
     -> /stream

You: /stream <Tab>
     -> on  off
```

## Differences from Original Client

| Feature | Original (`ai_client.py`) | Enhanced |
|---------|---------------------------|----------|
| Multiline input | No | Yes (Alt+Enter / Ctrl+J) |
| Autocomplete | No | Yes (Tab) |
| History search | No | Yes (Ctrl+R) |
| Raw output toggle | No | Yes (`/raw`) |
| Clipboard copy | No | Yes (`/copy`) |
| Configurable URL | Hardcoded | Env var / config file |

## Requirements

- Python 3.8+
- prompt-toolkit >= 3.0.40
- rich >= 13.0.0
- requests >= 2.28.0
- pyyaml >= 6.0 (optional, for config file support)
- pyperclip >= 1.8.0 (optional, for clipboard support)

## Troubleshooting

### Newline shortcuts not working

Different terminals handle key combinations differently. Try these alternatives:
- **Alt+Enter** - Hold Alt and press Enter
- **Escape, Enter** - Press Escape, release, then press Enter
- **Ctrl+J** - Most reliable across all terminals

### Clipboard not working

Install pyperclip: `pip install pyperclip`

On Linux, you may also need: `sudo apt install xclip` or `xsel`

### Config file not loading

Ensure you have PyYAML installed: `pip install pyyaml`
