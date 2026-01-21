"""
ai_client_enhanced.py - Enhanced AI Lab Client

Features:
- Multiline input: Shift+Enter (Windows) or Escape+Enter (universal) for newlines
- Command autocomplete: Tab to complete /commands
- Better copy support: /raw toggle and /copy command
- Cross-platform compatibility
- Configurable server URL via environment variable or config file
"""

import requests
from typing import List, Dict, Optional
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.live import Live
from rich import box
from pathlib import Path
import json
from datetime import datetime

# prompt_toolkit imports
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.history import FileHistory
from prompt_toolkit.formatted_text import HTML

# Local config
from config import load_config

# Initialize Rich console
console = Console()


class SlashCommandCompleter(Completer):
    """Autocomplete for slash commands with subcommand support."""

    COMMANDS = {
        "/exit": {"_desc": "Exit the client"},
        "/quit": {"_desc": "Exit the client"},
        "/reset": {"_desc": "Clear conversation history"},
        "/model": {
            "_desc": "Show and select models",
            "info": "Show detailed model information",
        },
        "/models": {
            "_desc": "Show and select models",
            "info": "Show detailed model information",
        },
        "/export": {"_desc": "Export conversation to text file"},
        "/export-json": {"_desc": "Export conversation to JSON"},
        "/export-ft": {"_desc": "Export for fine-tuning (JSONL)"},
        "/help": {"_desc": "Show help message"},
        "/status": {"_desc": "Show server status"},
        "/hardware": {"_desc": "Show hardware configuration"},
        "/stream": {
            "_desc": "Toggle streaming mode",
            "on": "Enable streaming (token-by-token)",
            "off": "Disable streaming (batch mode)",
        },
        "/system": {
            "_desc": "Manage system prompt",
            "reset": "Reset to default system prompt",
        },
        "/layers": {"_desc": "Get/set GPU layer count"},
        "/mem": {"_desc": "Show memory usage"},
        "/tools": {
            "_desc": "Manage tools",
            "on": "Enable tool calling",
            "off": "Disable tool calling",
        },
        "/stop-server": {"_desc": "Shutdown the server"},
        "/raw": {"_desc": "Toggle raw output mode (for copy/paste)"},
        "/copy": {"_desc": "Copy last response to clipboard"},
        "/cd": {"_desc": "Set working directory for file operations"},
        "/pwd": {"_desc": "Show current working directory"},
        "/ls": {"_desc": "List files in working directory"},
    }

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        words = text.split()

        # Complete command name
        if text.startswith("/") and len(words) <= 1:
            word = text
            for cmd, subcommands in self.COMMANDS.items():
                if cmd.startswith(word):
                    desc = subcommands.get("_desc", "")
                    yield Completion(
                        cmd,
                        start_position=-len(word),
                        display_meta=desc,
                    )

        # Complete subcommand
        elif len(words) >= 1 and words[0] in self.COMMANDS:
            cmd = words[0]
            subcommands = self.COMMANDS[cmd]
            partial = words[1] if len(words) > 1 else ""

            for sub, desc in subcommands.items():
                if sub.startswith("_"):
                    continue
                if sub.startswith(partial):
                    yield Completion(
                        sub,
                        start_position=-len(partial),
                        display_meta=desc,
                    )


class ChatClient:
    """Enhanced chat client with prompt_toolkit integration."""

    def __init__(self, server_url: str, history_file: str):
        self.server_url = server_url
        self.conversation_history: List[Dict[str, str]] = []
        self.system_prompt: Optional[str] = None
        self.streaming_enabled: bool = True
        self.raw_output: bool = False
        self.last_response: str = ""
        self.working_directory: Optional[Path] = None  # Working directory for file ops
        self.input_session = self._create_input_session(history_file)

    def _create_input_session(self, history_file: str) -> PromptSession:
        """Create prompt_toolkit session with custom key bindings."""
        bindings = KeyBindings()

        @bindings.add(Keys.Enter)
        def submit(event):
            """Enter submits the input."""
            event.current_buffer.validate_and_handle()

        @bindings.add("escape", "enter")  # Alt+Enter or Escape then Enter
        def newline_alt(event):
            """Alt+Enter or Escape+Enter inserts a newline."""
            event.current_buffer.insert_text("\n")

        @bindings.add("c-j")  # Ctrl+J (alternative for newline)
        def newline_ctrl_j(event):
            """Ctrl+J inserts a newline."""
            event.current_buffer.insert_text("\n")

        return PromptSession(
            completer=SlashCommandCompleter(),
            key_bindings=bindings,
            multiline=True,
            prompt_continuation="... ",  # Show continuation indicator
            complete_while_typing=True,
            enable_history_search=True,  # Ctrl+R to search
            history=FileHistory(history_file),
        )

    def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: dict = None,
        timeout: int = 30,
        error_prefix: str = "Request",
    ) -> Optional[Dict]:
        """Generic request handler with error handling."""
        try:
            url = f"{self.server_url}{endpoint}"
            response = getattr(requests, method)(url, json=json_data, timeout=timeout)

            if response.status_code == 200:
                return response.json()
            else:
                error = response.json().get("detail", "Unknown error")
                console.print(f"[red]{error_prefix} error: {error}[/red]")
                return None
        except requests.exceptions.Timeout:
            console.print(f"[red]{error_prefix} timed out[/red]")
            return None
        except requests.exceptions.RequestException as e:
            console.print(f"[red]{error_prefix} failed: {e}[/red]")
            return None

    def check_server_health(self) -> bool:
        """Check if server is healthy."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                return health.get("model_loaded", False)
            return False
        except requests.exceptions.RequestException:
            return False

    def get_models_list(self) -> Optional[Dict]:
        """Get list of available models from server."""
        try:
            response = requests.get(f"{self.server_url}/models", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except requests.exceptions.RequestException:
            return None

    def get_hardware(self) -> Optional[Dict]:
        """Get hardware information from server."""
        try:
            response = requests.get(f"{self.server_url}/hardware", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Failed to get hardware info: {e}[/red]")
            return None

    def switch_model(self, model_key: str) -> Optional[Dict]:
        """Switch to a different model (5 min timeout for downloads)."""
        try:
            response = requests.post(
                f"{self.server_url}/model/switch",
                json={"model_key": model_key},
                timeout=300,
            )

            if response.status_code == 200:
                return response.json()
            else:
                error = response.json().get("detail", "Unknown error")
                console.print(f"[red]Error switching model: {error}[/red]")
                return None
        except requests.exceptions.Timeout:
            console.print(
                "[red]Model switch timed out (5 min). Check server logs.[/red]"
            )
            return None
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Failed to switch model: {e}[/red]")
            return None

    def _get_effective_system_prompt(self) -> Optional[str]:
        """Build system prompt including working directory context if set."""
        parts = []

        if self.system_prompt:
            parts.append(self.system_prompt)

        if self.working_directory:
            wd_context = (
                f"\n\nWORKING DIRECTORY: {self.working_directory}\n"
                "When performing file operations (read, write, create, list), use this directory as the base. "
                "All relative paths should be resolved relative to this directory. "
                "For absolute paths, use this directory unless the user specifies otherwise."
            )
            parts.append(wd_context)

        return "\n".join(parts) if parts else None

    def send_message(
        self, message: str, temperature: float = 0.7, max_tokens: int = 8192
    ) -> Optional[Dict]:
        """Send a message to the server (batch mode)."""
        self.conversation_history.append({"role": "user", "content": message})

        payload = {
            "messages": self.conversation_history,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        effective_prompt = self._get_effective_system_prompt()
        if effective_prompt:
            payload["system_prompt"] = effective_prompt

        try:
            response = requests.post(
                f"{self.server_url}/chat", json=payload, timeout=300
            )

            if response.status_code == 200:
                result = response.json()

                if result.get("approval_required"):
                    approval_result = self._handle_tool_approval(result)
                    if approval_result:
                        self.conversation_history.append(
                            {"role": "assistant", "content": approval_result["response"]}
                        )
                        return approval_result
                    else:
                        self.conversation_history.pop()
                        return None
                else:
                    self.conversation_history.append(
                        {"role": "assistant", "content": result["response"]}
                    )
                    return result

            elif response.status_code == 503:
                error_detail = response.json().get("detail", "Server busy")
                console.print(f"[yellow]{error_detail}[/yellow]")
                self.conversation_history.pop()
                return None
            else:
                error = response.json().get("detail", "Unknown error")
                console.print(f"[red]Error: {error}[/red]")
                self.conversation_history.pop()
                return None

        except requests.exceptions.Timeout:
            console.print("[red]Request timed out.[/red]")
            self.conversation_history.pop()
            return None
        except requests.exceptions.ConnectionError:
            console.print(f"[red]Cannot connect to server at {self.server_url}[/red]")
            self.conversation_history.pop()
            return None
        except Exception as e:
            console.print(f"[red]Unexpected error: {e}[/red]")
            self.conversation_history.pop()
            return None

    def _handle_tool_approval(self, approval_response: Dict) -> Optional[Dict]:
        """Handle tool approval flow."""
        tools_pending = approval_response.get("tools_pending", [])
        message = approval_response.get("message", "")

        console.print(f"\n[yellow bold]Tool Approval Required[/yellow bold]")
        console.print(f"[yellow]{message}[/yellow]\n")

        table = Table(title="Tools Requesting Approval", box=box.ROUNDED)
        table.add_column("Tool", style="cyan", no_wrap=True)
        table.add_column("Arguments", style="white")

        for tool in tools_pending:
            tool_name = tool.get("tool_name", "unknown")
            arguments = tool.get("arguments", {})
            args_str = json.dumps(arguments, indent=2)
            table.add_row(tool_name, args_str)

        console.print(table)
        console.print()

        decisions = []
        for tool in tools_pending:
            tool_name = tool.get("tool_name", "unknown")
            tool_call_id = tool.get("tool_call_id", "")
            arguments = tool.get("arguments", {})
            args_display = ", ".join([f"{k}={v}" for k, v in arguments.items()])

            prompt_text = f"Approve [cyan]{tool_name}[/cyan]({args_display})?"
            response = Prompt.ask(
                prompt_text,
                choices=["y", "n", "a", "d"],
                default="y",
                show_choices=True,
                console=console,
            )

            if response.lower() == "a":
                for remaining_tool in tools_pending[len(decisions) :]:
                    decisions.append(
                        {
                            "tool_call_id": remaining_tool.get("tool_call_id", ""),
                            "approved": True,
                        }
                    )
                break
            elif response.lower() == "d":
                for remaining_tool in tools_pending[len(decisions) :]:
                    decisions.append(
                        {
                            "tool_call_id": remaining_tool.get("tool_call_id", ""),
                            "approved": False,
                        }
                    )
                break
            else:
                approved = response.lower() == "y"
                decisions.append({"tool_call_id": tool_call_id, "approved": approved})

        try:
            console.print("\n[dim]Sending approval decisions...[/dim]")
            approval_payload = {"decisions": decisions}
            response = requests.post(
                f"{self.server_url}/chat/approve",
                json=approval_payload,
                timeout=300,
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("approval_required"):
                    return self._handle_tool_approval(result)
                else:
                    return result
            else:
                error = response.json().get("detail", "Unknown error")
                console.print(f"[red]Error processing approval: {error}[/red]")
                return None

        except requests.exceptions.Timeout:
            console.print("[red]Approval request timed out.[/red]")
            return None
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Failed to send approval: {e}[/red]")
            return None

    def send_message_streaming(
        self, message: str, temperature: float = 0.7, max_tokens: int = 8192
    ) -> Optional[Dict]:
        """Send a message with streaming response."""
        self.conversation_history.append({"role": "user", "content": message})

        payload = {
            "messages": self.conversation_history,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        effective_prompt = self._get_effective_system_prompt()
        if effective_prompt:
            payload["system_prompt"] = effective_prompt

        try:
            accumulated_response = ""
            metadata = {}

            with requests.post(
                f"{self.server_url}/chat/stream",
                json=payload,
                stream=True,
                timeout=300,
            ) as response:
                if response.status_code != 200:
                    try:
                        error = response.json().get("detail", "Unknown error")
                    except:
                        error = f"HTTP {response.status_code}"
                    console.print(f"[red]Error: {error}[/red]")
                    self.conversation_history.pop()
                    return None

                with Live(
                    Panel(
                        "", title="[bold green]Assistant[/bold green]", border_style="green"
                    ),
                    console=console,
                    refresh_per_second=30,
                    transient=True,
                ) as live:
                    for line in response.iter_lines():
                        if not line:
                            continue

                        line = line.decode("utf-8")
                        if not line.startswith("data: "):
                            continue

                        data_str = line[6:]

                        if data_str == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)

                            if "error" in chunk:
                                console.print(f"[red]Error: {chunk['error']}[/red]")
                                self.conversation_history.pop()
                                return None

                            if chunk.get("type") == "stream_end":
                                metadata = chunk
                                continue

                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            if "content" in delta and delta["content"] is not None:
                                token = delta["content"]
                                accumulated_response += token

                                live.update(
                                    Panel(
                                        accumulated_response,
                                        title="[bold green]Assistant[/bold green]",
                                        border_style="green",
                                    )
                                )

                        except json.JSONDecodeError:
                            continue

            # Store last response for /copy
            self.last_response = accumulated_response

            # Show final panel
            self._display_response(accumulated_response)

            self.conversation_history.append(
                {"role": "assistant", "content": accumulated_response}
            )

            return {
                "response": accumulated_response,
                "tokens_input": metadata.get("tokens_input", 0),
                "tokens_generated": metadata.get("tokens_generated", 0),
                "tokens_total": metadata.get("tokens_total", 0),
                "generation_time": metadata.get("generation_time", 0),
                "tokens_per_second": metadata.get("tokens_per_second", 0),
                "tools_used": metadata.get("tools_used"),
                "device": metadata.get("device", "Unknown"),
            }

        except requests.exceptions.Timeout:
            console.print("[red]Request timed out.[/red]")
            self.conversation_history.pop()
            return None
        except requests.exceptions.ConnectionError:
            console.print(f"[red]Cannot connect to server at {self.server_url}[/red]")
            self.conversation_history.pop()
            return None
        except Exception as e:
            console.print(f"[red]Streaming error: {e}[/red]")
            self.conversation_history.pop()
            return None

    def _display_response(self, response_text: str):
        """Display response based on raw_output setting."""
        if self.raw_output:
            # Plain text output for easy copy/paste
            console.print("\n[bold green]Assistant:[/bold green]")
            console.print(response_text)
            console.print()
        else:
            # Panel output (default)
            console.print(
                Panel(
                    response_text,
                    title="[bold green]Assistant[/bold green]",
                    border_style="green",
                    box=box.ROUNDED,
                )
            )

    def _list_directory(self, path: Path, brief: bool = False):
        """List contents of a directory."""
        try:
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))

            if brief:
                # Brief listing - just counts
                dirs = sum(1 for i in items if i.is_dir())
                files = sum(1 for i in items if i.is_file())
                console.print(f"[dim]Contains: {dirs} folders, {files} files[/dim]")
            else:
                # Full listing with table
                table = Table(
                    title=f"Contents of {path}",
                    box=box.ROUNDED,
                    show_header=True,
                    header_style="bold cyan",
                )
                table.add_column("Type", style="dim", width=6)
                table.add_column("Name", style="cyan")
                table.add_column("Size", justify="right", width=12)

                for item in items[:50]:  # Limit to 50 items
                    if item.is_dir():
                        table.add_row("[DIR]", f"[bold blue]{item.name}/[/bold blue]", "")
                    else:
                        size = item.stat().st_size
                        if size < 1024:
                            size_str = f"{size} B"
                        elif size < 1024 * 1024:
                            size_str = f"{size / 1024:.1f} KB"
                        else:
                            size_str = f"{size / (1024 * 1024):.1f} MB"
                        table.add_row("[FILE]", item.name, size_str)

                console.print(table)

                if len(items) > 50:
                    console.print(f"[dim]... and {len(items) - 50} more items[/dim]")

        except PermissionError:
            console.print(f"[red]Permission denied: {path}[/red]")
        except Exception as e:
            console.print(f"[red]Error listing directory: {e}[/red]")

    def send_command(
        self, command: str, value: Optional[str] = None
    ) -> Optional[Dict]:
        """Send a command to the server."""
        payload = {"command": command}
        if value is not None:
            payload["value"] = value
        return self._make_request(
            "post", "/command", json_data=payload, error_prefix="Command"
        )

    def shutdown_server(self) -> bool:
        """Request server shutdown."""
        try:
            response = requests.post(f"{self.server_url}/shutdown", timeout=5)
            return response.status_code == 200
        except:
            return False

    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
        console.print("[yellow]Conversation history cleared.[/yellow]")

    def show_models_info(self):
        """Show detailed information about all models."""
        console.print("\n[yellow]Fetching model information...[/yellow]")
        models_data = self.get_models_list()

        if not models_data:
            console.print("[red]Failed to retrieve models list.[/red]")
            return

        models = models_data.get("models", [])
        console.print("\n")

        for model in models:
            if not model.get("exists", True):
                status_marker = "NOT FOUND"
                status_color = "red"
            elif model["is_current"]:
                status_marker = "CURRENT"
                status_color = "green"
            elif model["recommended"]:
                status_marker = "RECOMMENDED"
                status_color = "yellow"
            else:
                status_marker = ""
                status_color = "cyan"

            header = f"[bold {status_color}]{model['key']}[/bold {status_color}]"
            if status_marker:
                header += f" [{status_color}]({status_marker})[/{status_color}]"

            info = []
            info.append(f"[bold]Model:[/bold] {model['name']}")
            info.append(f"[bold]Description:[/bold] {model['description']}")
            info.append(
                f"[bold]Context Length:[/bold] {model['context_length']:,} tokens"
            )
            info.append(f"[bold]VRAM Estimate:[/bold] {model['vram_estimate']}")

            if not model.get("exists", True):
                info.append(f"\n[bold red]Status:[/bold red] Model file not found!")

            if "usage" in model and model.get("usage"):
                info.append(f"\n[bold yellow]Use Cases:[/bold yellow]\n{model['usage']}")

            console.print(
                Panel(
                    "\n".join(info),
                    title=header,
                    border_style=status_color,
                    box=box.ROUNDED,
                )
            )
            console.print()

    def show_models_and_select(self) -> bool:
        """Show available models and allow selection."""
        console.print("\n[yellow]Fetching available models...[/yellow]")
        models_data = self.get_models_list()

        if not models_data:
            console.print("[red]Failed to retrieve models list.[/red]")
            return True

        models = models_data.get("models", [])
        current_model_key = models_data.get("current_model_key", "")

        table = Table(
            title="Available Models",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Key", style="cyan", width=25)
        table.add_column("Status", width=14)
        table.add_column("Description", width=45)
        table.add_column("Context", width=10)
        table.add_column("VRAM", width=8)

        for idx, model in enumerate(models, 1):
            if not model.get("exists", True):
                status = "NOT FOUND"
                status_style = "red bold"
            elif model["is_current"]:
                status = "CURRENT"
                status_style = "green bold"
            elif model["recommended"]:
                status = "RECOMMENDED"
                status_style = "yellow"
            else:
                status = ""
                status_style = "dim"

            table.add_row(
                str(idx),
                model["key"],
                f"[{status_style}]{status}[/{status_style}]",
                model["description"],
                f"{model['context_length']//1000}k",
                model["vram_estimate"],
            )

        console.print("\n")
        console.print(table)
        console.print("\n")
        console.print(
            "[dim]Type number to select, 'info' for details, or 'cancel' to exit[/dim]"
        )

        while True:
            try:
                choice = (
                    Prompt.ask("[bold blue]Select model[/bold blue]", default="cancel")
                    .strip()
                    .lower()
                )

                if choice == "info":
                    self.show_models_info()
                    return True

                if choice in ["cancel", "c", "esc", ""]:
                    console.print("[yellow]Selection cancelled.[/yellow]")
                    return True

                try:
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(models):
                        selected = models[choice_num - 1]
                        model_key = selected["key"]

                        if not selected.get("exists", True):
                            console.print(f"[red]Model file not found![/red]")
                            continue

                        if model_key == current_model_key:
                            console.print(f"[yellow]Already using {model_key}[/yellow]")
                            return True

                        console.print(f"\n[cyan]Selected: {selected['name']}[/cyan]")
                        console.print(f"[dim]{selected['description']}[/dim]")

                        if selected.get("usage"):
                            console.print(
                                f"\n[yellow]Use cases:[/yellow] {selected['usage']}"
                            )

                        confirm = Prompt.ask(
                            "\n[yellow]Switch to this model?[/yellow]",
                            choices=["y", "n"],
                            default="y",
                        )

                        if confirm.lower() == "y":
                            console.print(f"\n[yellow]Switching to {model_key}...[/yellow]")
                            console.print(
                                "[dim]This may take 10-30 seconds...[/dim]\n"
                            )

                            result = self.switch_model(model_key)

                            if result and result.get("status") == "success":
                                console.print(
                                    f"[green]Successfully switched to {model_key}![/green]"
                                )
                                console.print(
                                    f"[dim]{result['previous_model']} -> {result['new_model']}[/dim]"
                                )

                                if self.conversation_history:
                                    clear = Prompt.ask(
                                        "\n[yellow]Clear conversation history?[/yellow]",
                                        choices=["y", "n"],
                                        default="y",
                                    )
                                    if clear.lower() == "y":
                                        self.reset_conversation()

                            return True
                        else:
                            console.print("[yellow]Switch cancelled.[/yellow]")
                            return True
                    else:
                        console.print(
                            f"[red]Invalid selection. Choose 1-{len(models)}[/red]"
                        )

                except ValueError:
                    matching = [m for m in models if m["key"] == choice]
                    if matching:
                        if not matching[0].get("exists", True):
                            console.print(f"[red]Model file not found![/red]")
                            continue
                        if matching[0]["key"] == current_model_key:
                            console.print(
                                f"[yellow]Already using {matching[0]['key']}[/yellow]"
                            )
                            return True
                        result = self.switch_model(matching[0]["key"])
                        if result and result.get("status") == "success":
                            console.print(
                                f"[green]Successfully switched to {matching[0]['key']}![/green]"
                            )
                        return True
                    console.print(f"[red]Unknown model: {choice}[/red]")

            except KeyboardInterrupt:
                console.print("\n[yellow]Selection cancelled.[/yellow]")
                return True

    def handle_slash_command(self, command: str) -> bool:
        """Handle slash commands. Returns False to exit, True to continue."""
        parts = command[1:].split(maxsplit=1)
        cmd = parts[0].lower()
        value = parts[1] if len(parts) > 1 else None

        if cmd in ["exit", "quit"]:
            return False

        elif cmd == "reset":
            self.reset_conversation()
            return True

        elif cmd in ["model", "models"]:
            if value and value.lower() == "info":
                self.show_models_info()
            else:
                return self.show_models_and_select()
            return True

        elif cmd == "export":
            if value is None:
                console.print("[yellow]Usage: /export <filepath>[/yellow]")
            else:
                self.export_conversation(value)
            return True

        elif cmd == "export-json":
            if value is None:
                console.print("[yellow]Usage: /export-json <filepath>[/yellow]")
            else:
                self.export_conversation_json(value)
            return True

        elif cmd == "export-ft":
            if value is None:
                console.print("[yellow]Usage: /export-ft <filepath>[/yellow]")
            else:
                self.export_for_finetuning(value)
            return True

        elif cmd == "help":
            self.show_help()
            return True

        elif cmd == "status":
            self.show_status()
            return True

        elif cmd == "hardware":
            self.show_hardware()
            return True

        elif cmd == "stream":
            if value is None:
                status = (
                    "[green]enabled[/green]"
                    if self.streaming_enabled
                    else "[yellow]disabled[/yellow]"
                )
                console.print(f"[cyan]Streaming: {status}[/cyan]")
            elif value.lower() in ["on", "enable", "enabled", "true", "1"]:
                self.streaming_enabled = True
                console.print(
                    "[green]Streaming enabled - tokens will appear as generated[/green]"
                )
            elif value.lower() in ["off", "disable", "disabled", "false", "0"]:
                self.streaming_enabled = False
                console.print("[yellow]Streaming disabled - using batch mode[/yellow]")
            else:
                console.print("[red]Usage: /stream [on|off][/red]")
            return True

        elif cmd == "raw":
            self.raw_output = not self.raw_output
            if self.raw_output:
                console.print(
                    "[green]Raw output mode enabled - responses shown as plain text for easy copy[/green]"
                )
            else:
                console.print(
                    "[yellow]Raw output mode disabled - responses shown in panels[/yellow]"
                )
            return True

        elif cmd == "copy":
            if not self.last_response:
                console.print("[yellow]No response to copy yet.[/yellow]")
            else:
                # Try to copy to clipboard
                try:
                    import pyperclip

                    pyperclip.copy(self.last_response)
                    console.print("[green]Last response copied to clipboard![/green]")
                except ImportError:
                    # Fallback: print the raw text
                    console.print(
                        "[yellow]pyperclip not installed. Here's the raw response:[/yellow]\n"
                    )
                    console.print(self.last_response)
                    console.print(
                        "\n[dim]Install pyperclip for clipboard support: pip install pyperclip[/dim]"
                    )
            return True

        elif cmd == "cd":
            if value is None:
                if self.working_directory:
                    console.print(f"[cyan]Working directory: {self.working_directory}[/cyan]")
                else:
                    console.print("[yellow]No working directory set. Use /cd <path> to set one.[/yellow]")
            else:
                # Expand ~ and resolve path
                try:
                    new_path = Path(value).expanduser().resolve()
                    if new_path.exists():
                        if new_path.is_dir():
                            self.working_directory = new_path
                            console.print(f"[green]Working directory set to: {new_path}[/green]")
                            # Show directory contents
                            self._list_directory(new_path, brief=True)
                        else:
                            console.print(f"[red]Not a directory: {new_path}[/red]")
                    else:
                        console.print(f"[red]Directory does not exist: {new_path}[/red]")
                except Exception as e:
                    console.print(f"[red]Error setting directory: {e}[/red]")
            return True

        elif cmd == "pwd":
            if self.working_directory:
                console.print(f"[cyan]Working directory: {self.working_directory}[/cyan]")
            else:
                console.print("[yellow]No working directory set. Use /cd <path> to set one.[/yellow]")
            return True

        elif cmd == "ls":
            if self.working_directory:
                self._list_directory(self.working_directory, brief=False)
            else:
                console.print("[yellow]No working directory set. Use /cd <path> first.[/yellow]")
            return True

        elif cmd == "system":
            if value is None:
                result = self.send_command("system")
                if result:
                    console.print(
                        Panel(
                            result["current_value"],
                            title="System Prompt",
                            border_style="cyan",
                        )
                    )
            elif value.lower() == "reset":
                result = self.send_command("system", "reset")
                if result:
                    console.print(f"[green]{result['message']}[/green]")
                    self.system_prompt = None
            else:
                result = self.send_command("system", value)
                if result:
                    console.print(f"[green]{result['message']}[/green]")
                    self.system_prompt = value
            return True

        elif cmd == "layers":
            if value is None:
                result = self.send_command("layers")
                if result:
                    console.print(f"[cyan]GPU Layers: {result['current_value']}[/cyan]")
            else:
                result = self.send_command("layers", value)
                if result:
                    console.print(f"[green]{result['message']}[/green]")
            return True

        elif cmd == "mem":
            result = self.send_command("mem")
            if result:
                console.print(f"[cyan]{result['current_value']}[/cyan]")
            return True

        elif cmd == "tools":
            if value is None:
                result = self.send_command("tools")
                if result:
                    console.print(f"[cyan]Tools: {result['current_value']}[/cyan]")
            else:
                result = self.send_command("tools", value)
                if result:
                    console.print(f"[green]{result['message']}[/green]")
            return True

        elif cmd == "stop-server":
            console.print("[yellow]Requesting server shutdown...[/yellow]")
            if self.shutdown_server():
                console.print("[green]Server shutdown requested.[/green]")
            else:
                console.print("[red]Failed to shutdown server.[/red]")
            return False

        else:
            console.print(f"[red]Unknown command: /{cmd}[/red]")
            console.print("[yellow]Type /help for available commands[/yellow]")
            return True

    def export_conversation(self, filepath: str):
        """Export conversation to text file."""
        if not self.conversation_history:
            console.print("[yellow]No conversation to export.[/yellow]")
            return

        try:
            path = Path(filepath)
            if path.is_dir():
                path = path / "conversation.txt"
            elif path.suffix.lower() != ".txt":
                path = path.with_suffix(".txt")

            path.parent.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            content = [
                "=" * 80,
                "AI Lab Conversation Export (Enhanced Client)",
                f"Exported: {timestamp}",
                f"Messages: {len(self.conversation_history)}",
            ]

            if self.system_prompt:
                content.append(f"System Prompt: {self.system_prompt}")

            content.extend(["=" * 80, ""])

            for i, msg in enumerate(self.conversation_history, 1):
                role = msg["role"].upper()
                content.extend([f"[{i}] {role}:", "-" * 80, msg["content"], ""])

            content.extend(["=" * 80, "End of conversation", "=" * 80])

            path.write_text("\n".join(content), encoding="utf-8")
            console.print(f"[green]Exported to: {path.absolute()}[/green]")
            console.print(f"[dim]({len(self.conversation_history)} messages)[/dim]")

        except PermissionError:
            console.print(f"[red]Permission denied: {filepath}[/red]")
        except Exception as e:
            console.print(f"[red]Export error: {e}[/red]")

    def export_conversation_json(self, filepath: str):
        """Export conversation to JSON."""
        if not self.conversation_history:
            console.print("[yellow]No conversation to export.[/yellow]")
            return

        try:
            path = Path(filepath)
            if path.is_dir():
                path = path / "conversation.json"
            elif path.suffix.lower() != ".json":
                path = path.with_suffix(".json")

            path.parent.mkdir(parents=True, exist_ok=True)

            export_data = {
                "exported_at": datetime.now().isoformat(),
                "message_count": len(self.conversation_history),
                "system_prompt": self.system_prompt,
                "messages": self.conversation_history,
            }

            path.write_text(json.dumps(export_data, indent=2), encoding="utf-8")
            console.print(f"[green]Exported to: {path.absolute()}[/green]")

        except Exception as e:
            console.print(f"[red]Export error: {e}[/red]")

    def export_for_finetuning(self, filepath: str):
        """Export in JSONL format for fine-tuning."""
        if not self.conversation_history:
            console.print("[yellow]No conversation to export.[/yellow]")
            return

        try:
            path = Path(filepath)
            if path.is_dir():
                path = path / "finetuning_data.jsonl"
            elif path.suffix.lower() not in [".jsonl", ".json"]:
                path = path.with_suffix(".jsonl")

            path.parent.mkdir(parents=True, exist_ok=True)

            training_example = {"messages": []}

            if self.system_prompt:
                training_example["messages"].append(
                    {"role": "system", "content": self.system_prompt}
                )

            training_example["messages"].extend(self.conversation_history)

            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(training_example) + "\n")

            console.print(f"[green]Exported to: {path.absolute()}[/green]")
            console.print("[dim]Format: JSONL (ready for training)[/dim]")

        except Exception as e:
            console.print(f"[red]Export error: {e}[/red]")

    def show_help(self):
        """Show help message."""
        help_text = """
[bold cyan]Input:[/bold cyan]
  Enter           Submit message
  Alt+Enter       New line (or Esc then Enter)
  Ctrl+J          New line (alternative)
  Tab             Autocomplete command
  Ctrl+R          Search command history

[bold cyan]Client Commands:[/bold cyan]
  /help           Show this help
  /exit, /quit    Exit client
  /reset          Clear conversation history
  /model, /models Show and select models
  /models info    Detailed model information
  /export <path>  Export to text file
  /export-json    Export to JSON
  /export-ft      Export for fine-tuning (JSONL)
  /status         Show server status
  /hardware       Show hardware configuration
  /stream [on|off] Toggle streaming mode
  /raw            Toggle raw output (for copy/paste)
  /copy           Copy last response to clipboard

[bold cyan]Working Directory:[/bold cyan]
  /cd <path>      Set working directory for file operations
  /cd             Show current working directory
  /pwd            Show current working directory
  /ls             List files in working directory

[bold cyan]Server Commands:[/bold cyan]
  /system         Show/set system prompt
  /system reset   Reset to default
  /layers         Get/set GPU layer count
  /tools [on|off] Enable/disable tools
  /mem            Show memory usage
  /stop-server    Shutdown server
"""
        console.print(
            Panel(help_text, title="Enhanced AI Client Help", border_style="cyan", box=box.ROUNDED)
        )

    def show_status(self):
        """Show current status."""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()

                ctx_percent = health.get("context_used_percent", 0)
                ctx_bar_width = 20
                filled = int(ctx_percent / 100 * ctx_bar_width)
                ctx_bar = "[" + ("=" * filled) + (" " * (ctx_bar_width - filled)) + "]"

                if ctx_percent < 50:
                    ctx_color = "green"
                elif ctx_percent < 70:
                    ctx_color = "yellow"
                else:
                    ctx_color = "red"

                status_text = f"""
[bold cyan]Server Status:[/bold cyan]
  Status: [green]{health['status']}[/green]
  Backend: llama.cpp
  Model: {health['model_key']}
  Full Name: {health['model_name']}
  Device: {health['device']}
  GPU Layers: {health.get('n_gpu_layers', 'N/A')}
  Model Loaded: {'Yes' if health['model_loaded'] else 'No'}
  Generating: {'Yes' if health['is_generating'] else 'No'}
  Tools: {'Enabled' if health.get('tools_enabled', True) else 'Disabled'}

[bold cyan]Context Management:[/bold cyan]
  Context Size: {health.get('context_size', 0):,} tokens
  History Used: [{ctx_color}]{health.get('context_used_tokens', 0):,}[/{ctx_color}] ({ctx_percent:.1f}%)
  Usage: [{ctx_color}]{ctx_bar}[/{ctx_color}]

[bold cyan]Client Status:[/bold cyan]
  Messages: {len(self.conversation_history)}
  Custom System Prompt: {'Yes' if self.system_prompt else 'No'}
  Streaming: {'Enabled' if self.streaming_enabled else 'Disabled'}
  Raw Output: {'Enabled' if self.raw_output else 'Disabled'}
  Working Directory: {self.working_directory or 'Not set'}
"""
                console.print(Panel(status_text, title="Status", border_style="cyan"))
            else:
                console.print("[red]Could not get server status[/red]")
        except:
            console.print("[red]Server unreachable[/red]")

    def show_hardware(self):
        """Show hardware configuration."""
        hw_info = self.get_hardware()

        if not hw_info:
            console.print("[red]Could not retrieve hardware information[/red]")
            return

        profile = hw_info.get("profile", {})
        current_config = hw_info.get("current_config", {})
        device_string = hw_info.get("device_string", "Unknown")

        gpu = profile.get("gpu", {})
        cpu = profile.get("cpu", {})
        memory = profile.get("memory", {})
        recommended = profile.get("recommended_config", {})

        hardware_text = f"""[bold cyan]System Type:[/bold cyan] {profile.get('system_type', 'unknown')}

[bold yellow]GPU:[/bold yellow]"""

        if gpu.get("has_gpu", False):
            hardware_text += f"""
  Name: [green]{gpu.get('name', 'Unknown')}[/green]
  VRAM: [green]{gpu.get('vram_gb', 0):.1f}GB[/green]
  CUDA: {gpu.get('cuda_version', 'unknown')}"""
        else:
            hardware_text += """
  [yellow]None detected (CPU-only mode)[/yellow]"""

        hardware_text += f"""

[bold yellow]CPU:[/bold yellow]
  Name: {cpu.get('name', 'Unknown')}
  Cores: {cpu.get('physical_cores', 'unknown')} physical / {cpu.get('logical_cores', 'unknown')} logical

[bold yellow]RAM:[/bold yellow]
  Total: {memory.get('total_gb', 0):.1f}GB
  Available: {memory.get('available_gb', 0):.1f}GB

[bold cyan]Current Configuration:[/bold cyan]
  Mode: [cyan]{recommended.get('mode', 'unknown')}[/cyan]
  Device: {device_string}
  GPU Layers: {current_config.get('n_gpu_layers', 'N/A')}
  Context Size: {current_config.get('ctx_size', 'N/A')} tokens
"""

        console.print(
            Panel(
                hardware_text,
                title=f"Hardware - {self.server_url}",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )

    def run(self):
        """Main client loop."""
        console.print(
            Panel(
                "[bold cyan]Enhanced AI Lab Client v1.0[/bold cyan]\n"
                f"Connected to: {self.server_url}\n"
                "Type /help for commands | Alt+Enter for new line",
                box=box.DOUBLE,
            )
        )

        console.print("\n[yellow]Checking server...[/yellow]")
        if not self.check_server_health():
            console.print("[red]Server not responding or model not loaded![/red]")
            console.print(f"[yellow]Ensure server is running at {self.server_url}[/yellow]")
            return

        console.print("[green]Server ready![/green]\n")

        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                console.print(f"[cyan]Model: {health['model_key']}[/cyan]")
                console.print(
                    f"[dim]GPU Layers: {health.get('n_gpu_layers', 'N/A')} | Device: {health['device']}[/dim]"
                )
                stream_status = (
                    "[green]on[/green]"
                    if self.streaming_enabled
                    else "[yellow]off[/yellow]"
                )
                console.print(
                    f"[dim]Streaming: {stream_status} | Tab to autocomplete commands[/dim]\n"
                )
        except:
            pass

        while True:
            try:
                # Use prompt_toolkit for input
                user_input = self.input_session.prompt(
                    HTML("<b><ansiblue>You: </ansiblue></b>")
                ).strip()

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    if not self.handle_slash_command(user_input):
                        break
                    continue

                if self.streaming_enabled:
                    console.print("\n[dim]Streaming...[/dim]")
                    result = self.send_message_streaming(user_input)

                    if result:
                        if result.get("tools_used"):
                            console.print(
                                f"[dim]Tools: {', '.join(result['tools_used'])}[/dim]"
                            )

                        console.print(
                            f"[dim]({result['tokens_input']} in -> {result['tokens_generated']} out -> "
                            f"{result['tokens_total']} total | "
                            f"{result['generation_time']:.2f}s @ {result['tokens_per_second']:.1f} tok/s | {result['device']})[/dim]"
                        )
                else:
                    console.print("\n[dim]Generating...[/dim]")
                    result = self.send_message(user_input)

                    if result:
                        self.last_response = result["response"]
                        self._display_response(result["response"])

                        if result.get("tools_used"):
                            console.print(
                                f"[dim]Tools: {', '.join(result['tools_used'])}[/dim]"
                            )

                        console.print(
                            f"[dim]({result['tokens_input']} in -> {result['tokens_generated']} out -> "
                            f"{result['tokens_total']} total | "
                            f"{result['generation_time']:.2f}s @ {result['tokens_per_second']:.1f} tok/s | {result['device']})[/dim]"
                        )

            except KeyboardInterrupt:
                console.print("\n\n[yellow]Interrupted. Type /exit to quit.[/yellow]")
                continue
            except EOFError:
                break

        console.print("\n[cyan]Goodbye![/cyan]")


def main():
    config = load_config()
    client = ChatClient(config["server_url"], config["history_file"])
    client.run()


if __name__ == "__main__":
    main()
