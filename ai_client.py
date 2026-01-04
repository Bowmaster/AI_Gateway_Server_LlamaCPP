"""
ai_client.py - AI Lab Client (llama.cpp edition)
Connects to AI Lab Server and provides interactive chat interface
"""

import requests
from typing import List, Dict, Optional
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich import box
from pathlib import Path
import json
from datetime import datetime

# Configuration
SERVER_URL = "http://ailab.local:8080"

# Initialize Rich console for colored output
console = Console()

class ChatClient:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.conversation_history: List[Dict[str, str]] = []
        self.system_prompt: Optional[str] = None
        
    def check_server_health(self) -> bool:
        """Check if server is healthy"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                return health.get("model_loaded", False)
            return False
        except requests.exceptions.RequestException:
            return False
    
    def get_models_list(self) -> Optional[Dict]:
        """Get list of available models from server"""
        try:
            response = requests.get(f"{self.server_url}/models", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except requests.exceptions.RequestException:
            return None
    
    def switch_model(self, model_key: str) -> Optional[Dict]:
        """Switch to a different model"""
        try:
            response = requests.post(
                f"{self.server_url}/model/switch",
                json={"model_key": model_key},
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error = response.json().get("detail", "Unknown error")
                console.print(f"[red]Error switching model: {error}[/red]")
                return None
        except requests.exceptions.Timeout:
            console.print("[red]Model switch timed out.[/red]")
            return None
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Failed to switch model: {e}[/red]")
            return None
    
    def send_message(self, message: str, temperature: float = 0.7, max_tokens: int = 8192) -> Optional[Dict]:
        """Send a message to the server"""
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })
        
        # Prepare request
        payload = {
            "messages": self.conversation_history,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if self.system_prompt:
            payload["system_prompt"] = self.system_prompt
        
        try:
            response = requests.post(
                f"{self.server_url}/chat",
                json=payload,
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Add assistant response to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": result["response"]
                })
                
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
    
    def send_command(self, command: str, value: Optional[str] = None) -> Optional[Dict]:
        """Send a command to the server"""
        try:
            payload = {"command": command}
            if value is not None:
                payload["value"] = value
            
            response = requests.post(
                f"{self.server_url}/command",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error = response.json().get("detail", "Unknown error")
                console.print(f"[red]Command error: {error}[/red]")
                return None
                
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Command failed: {e}[/red]")
            return None
    
    def shutdown_server(self) -> bool:
        """Request server shutdown"""
        try:
            response = requests.post(f"{self.server_url}/shutdown", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        console.print("[yellow]Conversation history cleared.[/yellow]")
    
    def show_models_info(self):
        """Show detailed information about all models"""
        
        console.print("\n[yellow]Fetching model information...[/yellow]")
        models_data = self.get_models_list()
        
        if not models_data:
            console.print("[red]Failed to retrieve models list.[/red]")
            return
        
        models = models_data.get("models", [])
        console.print("\n")
        
        for model in models:
            # Determine status
            if not model.get("exists", True):
                status_marker = "✗ NOT FOUND"
                status_color = "red"
            elif model["is_current"]:
                status_marker = "✓ CURRENT"
                status_color = "green"
            elif model["recommended"]:
                status_marker = "★ RECOMMENDED"
                status_color = "yellow"
            else:
                status_marker = ""
                status_color = "cyan"
            
            header = f"[bold {status_color}]{model['key']}[/bold {status_color}]"
            if status_marker:
                header += f" [{status_color}]({status_marker})[/{status_color}]"
            
            # Build info
            info = []
            info.append(f"[bold]Model:[/bold] {model['name']}")
            info.append(f"[bold]Description:[/bold] {model['description']}")
            info.append(f"[bold]Context Length:[/bold] {model['context_length']:,} tokens")
            info.append(f"[bold]VRAM Estimate:[/bold] {model['vram_estimate']}")
            
            if not model.get("exists", True):
                info.append(f"\n[bold red]Status:[/bold red] Model file not found!")
                info.append(f"[dim]Download from HuggingFace in GGUF format[/dim]")
            
            if 'usage' in model and model.get('usage'):
                info.append(f"\n[bold yellow]Use Cases:[/bold yellow]\n{model['usage']}")
            
            console.print(Panel(
                "\n".join(info),
                title=header,
                border_style=status_color,
                box=box.ROUNDED
            ))
            console.print()
    
    def show_models_and_select(self) -> bool:
        """Show available models and allow selection"""
        
        console.print("\n[yellow]Fetching available models...[/yellow]")
        models_data = self.get_models_list()
        
        if not models_data:
            console.print("[red]Failed to retrieve models list.[/red]")
            return True
        
        models = models_data.get("models", [])
        current_model_key = models_data.get("current_model_key", "")
        
        # Create table
        table = Table(title="Available Models", box=box.ROUNDED, show_header=True, header_style="bold cyan")
        table.add_column("#", style="dim", width=3)
        table.add_column("Key", style="cyan", width=25)
        table.add_column("Status", width=14)
        table.add_column("Description", width=45)
        table.add_column("Context", width=10)
        table.add_column("VRAM", width=8)
        
        # Add models
        for idx, model in enumerate(models, 1):
            if not model.get("exists", True):
                status = "✗ NOT FOUND"
                status_style = "red bold"
            elif model["is_current"]:
                status = "✓ CURRENT"
                status_style = "green bold"
            elif model["recommended"]:
                status = "★ RECOMMENDED"
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
                model["vram_estimate"]
            )
        
        console.print("\n")
        console.print(table)
        console.print("\n")
        console.print("[dim]Type number to select, 'info' for details, or 'cancel' to exit[/dim]")
        
        while True:
            try:
                choice = Prompt.ask("[bold blue]Select model[/bold blue]", default="cancel").strip().lower()
                
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
                            console.print(f"[yellow]Download GGUF format from HuggingFace[/yellow]")
                            continue
                        
                        if model_key == current_model_key:
                            console.print(f"[yellow]Already using {model_key}[/yellow]")
                            return True
                        
                        console.print(f"\n[cyan]Selected: {selected['name']}[/cyan]")
                        console.print(f"[dim]{selected['description']}[/dim]")
                        
                        if selected.get('usage'):
                            console.print(f"\n[yellow]Use cases:[/yellow] {selected['usage']}")
                        
                        confirm = Prompt.ask("\n[yellow]Switch to this model?[/yellow]", choices=["y", "n"], default="y")
                        
                        if confirm.lower() == "y":
                            console.print(f"\n[yellow]Switching to {model_key}...[/yellow]")
                            console.print("[dim]This may take 10-30 seconds...[/dim]\n")
                            
                            result = self.switch_model(model_key)
                            
                            if result and result.get("status") == "success":
                                console.print(f"[green]✓ Successfully switched to {model_key}![/green]")
                                console.print(f"[dim]{result['previous_model']} → {result['new_model']}[/dim]")
                                
                                if self.conversation_history:
                                    clear = Prompt.ask("\n[yellow]Clear conversation history?[/yellow]", choices=["y", "n"], default="y")
                                    if clear.lower() == "y":
                                        self.reset_conversation()
                            
                            return True
                        else:
                            console.print("[yellow]Switch cancelled.[/yellow]")
                            return True
                    else:
                        console.print(f"[red]Invalid selection. Choose 1-{len(models)}[/red]")
                
                except ValueError:
                    matching = [m for m in models if m["key"] == choice]
                    if matching:
                        model_key = matching[0]["key"]
                        if not matching[0].get("exists", True):
                            console.print(f"[red]Model file not found![/red]")
                            continue
                        if model_key == current_model_key:
                            console.print(f"[yellow]Already using {model_key}[/yellow]")
                            return True
                        
                        result = self.switch_model(model_key)
                        if result and result.get("status") == "success":
                            console.print(f"[green]✓ Switched to {model_key}![/green]")
                        return True
                    else:
                        console.print(f"[red]Unknown model: {choice}[/red]")
            
            except KeyboardInterrupt:
                console.print("\n[yellow]Selection cancelled.[/yellow]")
                return True
    
    def handle_slash_command(self, command: str) -> bool:
        """Handle slash commands"""
        
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
                console.print("[dim]Example: /export ./my_chat.txt[/dim]")
            else:
                self.export_conversation(value)
            return True
        
        elif cmd == "export-json":
            if value is None:
                console.print("[yellow]Usage: /export-json <filepath>[/yellow]")
                console.print("[dim]Example: /export-json ./my_chat.json[/dim]")
            else:
                self.export_conversation_json(value)
            return True
        
        elif cmd == "export-ft":
            if value is None:
                console.print("[yellow]Usage: /export-ft <filepath>[/yellow]")
                console.print("[dim]Example: /export-ft ./my_chat.jsonl[/dim]")
            else:
                self.export_for_finetuning(value)
            return True
        
        elif cmd == "help":
            self.show_help()
            return True
        
        elif cmd == "status":
            self.show_status()
            return True
        
        # Server commands
        elif cmd == "system":
            if value is None:
                result = self.send_command("system")
                if result:
                    console.print(Panel(result["current_value"], title="System Prompt", border_style="cyan"))
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
        """Export conversation to text file"""
        if not self.conversation_history:
            console.print("[yellow]No conversation to export.[/yellow]")
            return
        
        try:
            path = Path(filepath)
            if path.is_dir():
                path = path / "conversation.txt"
            elif path.suffix.lower() != '.txt':
                path = path.with_suffix('.txt')
            
            path.parent.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            content = [
                "=" * 80,
                "AI Lab Conversation Export (llama.cpp)",
                f"Exported: {timestamp}",
                f"Messages: {len(self.conversation_history)}",
            ]
            
            if self.system_prompt:
                content.append(f"System Prompt: {self.system_prompt}")
            
            content.extend(["=" * 80, ""])
            
            for i, msg in enumerate(self.conversation_history, 1):
                role = msg['role'].upper()
                content.extend([
                    f"[{i}] {role}:",
                    "-" * 80,
                    msg['content'],
                    ""
                ])
            
            content.extend(["=" * 80, "End of conversation", "=" * 80])
            
            path.write_text("\n".join(content), encoding='utf-8')
            console.print(f"[green]✓ Exported to: {path.absolute()}[/green]")
            console.print(f"[dim]({len(self.conversation_history)} messages)[/dim]")
            
        except PermissionError:
            console.print(f"[red]Permission denied: {filepath}[/red]")
        except Exception as e:
            console.print(f"[red]Export error: {e}[/red]")
    
    def export_conversation_json(self, filepath: str):
        """Export conversation to JSON"""
        if not self.conversation_history:
            console.print("[yellow]No conversation to export.[/yellow]")
            return
        
        try:
            path = Path(filepath)
            if path.is_dir():
                path = path / "conversation.json"
            elif path.suffix.lower() != '.json':
                path = path.with_suffix('.json')
            
            path.parent.mkdir(parents=True, exist_ok=True)
            
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "message_count": len(self.conversation_history),
                "system_prompt": self.system_prompt,
                "messages": self.conversation_history
            }
            
            path.write_text(json.dumps(export_data, indent=2), encoding='utf-8')
            console.print(f"[green]✓ Exported to: {path.absolute()}[/green]")
            
        except Exception as e:
            console.print(f"[red]Export error: {e}[/red]")
    
    def export_for_finetuning(self, filepath: str):
        """Export in JSONL format for fine-tuning"""
        if not self.conversation_history:
            console.print("[yellow]No conversation to export.[/yellow]")
            return
        
        try:
            path = Path(filepath)
            if path.is_dir():
                path = path / "finetuning_data.jsonl"
            elif path.suffix.lower() not in ['.jsonl', '.json']:
                path = path.with_suffix('.jsonl')
            
            path.parent.mkdir(parents=True, exist_ok=True)
            
            training_example = {"messages": []}
            
            if self.system_prompt:
                training_example["messages"].append({
                    "role": "system",
                    "content": self.system_prompt
                })
            
            training_example["messages"].extend(self.conversation_history)
            
            with open(path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(training_example) + '\n')
            
            console.print(f"[green]✓ Exported to: {path.absolute()}[/green]")
            console.print("[dim]Format: JSONL (ready for training)[/dim]")
            
        except Exception as e:
            console.print(f"[red]Export error: {e}[/red]")
    
    def show_help(self):
        """Show help message"""
        help_text = """
[bold cyan]Available Commands:[/bold cyan]

[bold]Client Commands:[/bold]
  /help                - Show this help
  /exit, /quit         - Exit client
  /reset               - Clear conversation history
  /model, /models      - Show and select models
  /models info         - Detailed model information
  /export <path>       - Export to text file
  /export-json <path>  - Export to JSON
  /export-ft <path>    - Export for fine-tuning (JSONL)
  /status              - Show server status

[bold]Server Commands:[/bold]
  /system              - Show system prompt
  /system <prompt>     - Set system prompt
  /system reset        - Reset to default
  /layers              - Show GPU layer count
  /layers <N>          - Set GPU layers (-1=all, 0=CPU)
  /tools               - Show tools status
  /tools <on|off>      - Enable/disable tools
  /mem                 - Show memory usage
  /stop-server         - Shutdown server

[bold cyan]llama.cpp Features:[/bold cyan]
  - Fast CUDA-accelerated inference
  - GGUF quantized models (Q4/Q5/Q8)
  - Adjustable GPU layer offloading
  - Large context windows (32k+ tokens)
"""
        console.print(Panel(help_text, title="Help", border_style="cyan", box=box.ROUNDED))
    
    def show_status(self):
        """Show current status"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                
                status_text = f"""
[bold cyan]Server Status:[/bold cyan]
  Status: [green]{health['status']}[/green]
  Backend: llama.cpp
  Model: {health['model_key']}
  Full Name: {health['model_name']}
  Device: {health['device']}
  GPU Layers: {health.get('n_gpu_layers', 'N/A')}
  Model Loaded: {'✓' if health['model_loaded'] else '✗'}
  Generating: {'Yes' if health['is_generating'] else 'No'}
  Tools: {'Enabled' if health.get('tools_enabled', True) else 'Disabled'}

[bold cyan]Client Status:[/bold cyan]
  Messages: {len(self.conversation_history)}
  Custom System Prompt: {'Yes' if self.system_prompt else 'No'}
"""
                console.print(Panel(status_text, title="Status", border_style="cyan"))
            else:
                console.print("[red]Could not get server status[/red]")
        except:
            console.print("[red]Server unreachable[/red]")
    
    def run(self):
        """Main client loop"""
        
        console.print(Panel(
            "[bold cyan]AI Lab Client v2.0 (llama.cpp)[/bold cyan]\n"
            f"Connected to: {self.server_url}\n"
            "Type /help for commands",
            box=box.DOUBLE
        ))
        
        console.print("\n[yellow]Checking server...[/yellow]")
        if not self.check_server_health():
            console.print("[red]Server not responding or model not loaded![/red]")
            console.print(f"[yellow]Ensure server is running at {self.server_url}[/yellow]")
            return
        
        console.print("[green]✓ Server ready![/green]\n")
        
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                console.print(f"[cyan]Model: {health['model_key']}[/cyan]")
                console.print(f"[dim]GPU Layers: {health.get('n_gpu_layers', 'N/A')} | Device: {health['device']}[/dim]")
                console.print("[dim]Type /model to switch | /help for commands[/dim]\n")
        except:
            pass
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold blue]You[/bold blue]").strip()
                
                if not user_input:
                    continue
                
                if user_input.startswith("/"):
                    if not self.handle_slash_command(user_input):
                        break
                    continue
                
                console.print("\n[dim]Generating...[/dim]")
                result = self.send_message(user_input)
                
                if result:
                    console.print(Panel(
                        result["response"],
                        title="[bold green]Assistant[/bold green]",
                        border_style="green",
                        box=box.ROUNDED
                    ))
                    
                    if result.get("tools_used"):
                        console.print(f"[dim]🔧 Tools: {', '.join(result['tools_used'])}[/dim]")
                    
                    console.print(
                        f"[dim]({result['tokens_input']} in → {result['tokens_generated']} out → "
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
    client = ChatClient(SERVER_URL)
    client.run()

if __name__ == "__main__":
    main()