"""
llama_manager.py - Updated with native HuggingFace download support
"""

import subprocess
import requests
import time
import signal
import os
import psutil
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class LlamaServerManager:
    """Manages the llama-server subprocess lifecycle with HuggingFace download support"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.server_url = f"http://{config['host']}:{config['port']}"
        self.is_running = False
        
        # Set up cache directory for HuggingFace downloads
        self.cache_dir = Path(config.get('cache_dir', './models'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def start(self, model_path_or_hf: str, use_hf: bool = False, ctx_size: Optional[int] = None) -> bool:
        """
        Start llama-server with the specified model.

        Args:
            model_path_or_hf: Either a local path OR HuggingFace repo:quantization string
            use_hf: If True, treat model_path_or_hf as HuggingFace identifier
            ctx_size: Override context size (if None, uses config default)

        Returns:
            True if started successfully, False otherwise
        """
        if self.is_running:
            logger.warning("llama-server is already running")
            return True

        # Validate executable exists
        executable = self.config['executable']
        if not os.path.exists(executable):
            logger.error(f"llama-server executable not found: {executable}")
            return False

        # Determine effective context size (parameter override or config default)
        effective_ctx_size = ctx_size if ctx_size is not None else self.config.get('ctx_size', 12288)
        self.effective_ctx_size = effective_ctx_size  # Store for retrieval

        # Build command line arguments
        cmd = [
            executable,
            "--host", self.config['host'],
            "--port", str(self.config['port']),
        ]

        # Add model - either local path or HuggingFace
        if use_hf:
            # HuggingFace download mode
            logger.info(f"Using HuggingFace model: {model_path_or_hf}")
            cmd.extend(["-hf", model_path_or_hf])
        else:
            # Local file mode
            if not os.path.exists(model_path_or_hf):
                logger.error(f"Model file not found: {model_path_or_hf}")
                return False
            cmd.extend(["--model", model_path_or_hf])

        # Add context size and GPU layers (using effective_ctx_size)
        cmd.extend([
            "--ctx-size", str(effective_ctx_size),
            "-n", str(self.config.get('n_predict', 8192)),
            "--n-gpu-layers", str(self.config.get('n_gpu_layers', -1)),
        ])

        # Add threads if specified
        if self.config.get('threads'):
            cmd.extend(["--threads", str(self.config['threads'])])

        # Add any additional args from config
        if self.config.get('additional_args'):
            cmd.extend(self.config['additional_args'])

        # Log hardware configuration being used
        logger.info("Hardware Configuration:")
        logger.info(f"  GPU Layers: {self.config.get('n_gpu_layers', -1)}")
        logger.info(f"  Context Size: {effective_ctx_size} tokens")
        logger.info(f"  CPU Threads: {self.config.get('threads', 'auto')}")

        logger.info(f"Starting llama-server with command: {' '.join(cmd)}")
        
        try:
            # Set environment variables
            env = os.environ.copy()
            
            # Set cache directory for HuggingFace downloads
            env['LLAMA_CACHE'] = str(self.cache_dir.absolute())
            logger.info(f"LLAMA_CACHE set to: {env['LLAMA_CACHE']}")
            
            # Start the process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0
            )
            
            # Wait for server to be ready (longer timeout for downloads)
            timeout = 300 if use_hf else 60  # 5 min for HF downloads, 1 min for local
            
            if self._wait_for_ready(timeout=timeout):
                self.is_running = True
                logger.info(f"llama-server started successfully (PID: {self.process.pid})")
                return True
            else:
                logger.error("llama-server failed to become ready")
                self.stop()
                return False
                
        except Exception as e:
            logger.error(f"Failed to start llama-server: {e}")
            return False

    def _monitor_stderr(self):
        """Monitor stderr for download progress and errors"""
        if not self.process or not self.process.stderr:
            return

        for line in self.process.stderr:
            line = line.strip()
            if not line:
                continue
            # Highlight download keywords
            level = "info" if any(kw in line.lower() for kw in ['download', 'fetching', 'progress', 'mb']) else "debug"
            getattr(logger, level)(f"llama-server: {line}")

    def _wait_for_ready(self, timeout: int = 60) -> bool:
        """
        Wait for llama-server to be ready by polling health endpoint.
        Monitors stderr for download progress messages.
        
        Args:
            timeout: Maximum seconds to wait
            
        Returns:
            True if server becomes ready, False if timeout
        """
        start_time = time.time()
        last_log_time = start_time
        
        # Monitor stderr for download progress
        import threading
        threading.Thread(target=self._monitor_stderr, daemon=True).start()
        
        while time.time() - start_time < timeout:
            try:
                # Check if process is still running
                if self.process and self.process.poll() is not None:
                    exit_code = self.process.returncode
                    logger.error(f"llama-server process terminated unexpectedly (exit code: {exit_code})")
                    # Capture any remaining stdout/stderr for diagnostics
                    try:
                        stdout_remaining = self.process.stdout.read() if self.process.stdout else ""
                        stderr_remaining = self.process.stderr.read() if self.process.stderr else ""
                        if stdout_remaining and stdout_remaining.strip():
                            logger.error(f"llama-server stdout:\n{stdout_remaining.strip()}")
                        if stderr_remaining and stderr_remaining.strip():
                            logger.error(f"llama-server stderr:\n{stderr_remaining.strip()}")
                        if not stdout_remaining.strip() and not stderr_remaining.strip():
                            logger.error("No output captured from llama-server - process may have crashed before producing output")
                    except Exception as e:
                        logger.error(f"Could not read llama-server output: {e}")
                    return False
                
                # Try to connect to health endpoint
                response = requests.get(f"{self.server_url}/health", timeout=2)
                if response.status_code == 200:
                    logger.info("llama-server health check passed")
                    
                    # Log actual context size being used
                    try:
                        props_response = requests.get(f"{self.server_url}/props", timeout=2)
                        if props_response.status_code == 200:
                            props = props_response.json()
                            actual_ctx = props.get('default_generation_settings', {}).get('n_ctx', 'unknown')
                            logger.info(f"Actual context size in use though this may be innacurate: {actual_ctx}")
                    except:
                        pass
                    
                    return True
                    
            except requests.exceptions.RequestException:
                # Server not ready yet, continue waiting
                pass
            
            # Log waiting status every 10 seconds for long downloads
            current_time = time.time()
            if current_time - last_log_time > 10:
                elapsed = int(current_time - start_time)
                logger.info(f"Waiting for llama-server to be ready... ({elapsed}s / {timeout}s)")
                last_log_time = current_time
            
            time.sleep(1)
        
        logger.error(f"llama-server did not become ready within {timeout} seconds")
        return False
    
    def stop(self) -> bool:
        """
        Stop the llama-server process gracefully.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.process:
            logger.warning("No llama-server process to stop")
            return True
        
        logger.info(f"Stopping llama-server (PID: {self.process.pid})")
        
        try:
            # Try graceful shutdown first
            if os.name == 'nt':
                # Windows: Send CTRL_BREAK_EVENT
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                # Unix: Send SIGTERM
                self.process.terminate()
            
            # Wait for process to exit
            try:
                self.process.wait(timeout=10)
                logger.info("llama-server stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't exit
                logger.warning("llama-server did not stop gracefully, force killing")
                self.process.kill()
                self.process.wait(timeout=5)
            
            self.is_running = False
            self.process = None
            return True
            
        except Exception as e:
            logger.error(f"Error stopping llama-server: {e}")
            return False
    
    def restart(self, model_path_or_hf: str, use_hf: bool = False, ctx_size: Optional[int] = None) -> bool:
        """
        Restart llama-server with a new model.

        Args:
            model_path_or_hf: Path to model or HuggingFace identifier
            use_hf: Whether to use HuggingFace download
            ctx_size: Override context size (if None, uses config default)

        Returns:
            True if restarted successfully, False otherwise
        """
        logger.info(f"Restarting llama-server with model: {model_path_or_hf}, ctx_size: {ctx_size}")

        if not self.stop():
            logger.error("Failed to stop llama-server for restart")
            return False

        # Brief pause to ensure clean shutdown
        time.sleep(1)

        return self.start(model_path_or_hf, use_hf=use_hf, ctx_size=ctx_size)
    
    def is_healthy(self) -> bool:
        """
        Check if llama-server is running and responsive.
        
        Returns:
            True if healthy, False otherwise
        """
        if not self.is_running or not self.process:
            return False
        
        # Check if process is still alive
        if self.process.poll() is not None:
            logger.warning("llama-server process has terminated")
            self.is_running = False
            return False
        
        # Check health endpoint
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get process statistics for llama-server.
        
        Returns:
            Dictionary with memory, CPU usage, etc.
        """
        if not self.process or not self.is_running:
            return {
                "running": False
            }
        
        try:
            proc = psutil.Process(self.process.pid)
            
            return {
                "running": True,
                "pid": self.process.pid,
                "memory_mb": proc.memory_info().rss / 1024 / 1024,
                "cpu_percent": proc.cpu_percent(interval=0.1),
                "status": proc.status(),
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {
                "running": False,
                "error": "Cannot access process information"
            }
    
    def __del__(self):
        """Cleanup: ensure process is stopped when manager is destroyed"""
        if self.is_running:
            logger.info("LlamaServerManager cleanup: stopping llama-server")
            self.stop()