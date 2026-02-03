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

        # Track last loaded model for idle reload
        self.last_model_identifier: Optional[str] = None
        self.last_use_hf: bool = False
        self.last_ctx_size: Optional[int] = None
        self.idle_unloaded: bool = False

        # Crash diagnostics — populated when process dies unexpectedly
        self.last_crash_info: Optional[Dict[str, Any]] = None
        
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

        # Remember model parameters for potential idle reload
        self.last_model_identifier = model_path_or_hf
        self.last_use_hf = use_hf
        self.last_ctx_size = ctx_size

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

        # CPU optimization flags (only meaningful for CPU/hybrid modes)
        if self.config.get('numa_mode'):
            cmd.extend(["--numa", str(self.config['numa_mode'])])
        if self.config.get('batch_size'):
            cmd.extend(["--batch-size", str(self.config['batch_size'])])
        if self.config.get('ubatch_size'):
            cmd.extend(["--ubatch-size", str(self.config['ubatch_size'])])
        if self.config.get('mlock'):
            cmd.append("--mlock")
        if self.config.get('threads_batch'):
            cmd.extend(["--threads-batch", str(self.config['threads_batch'])])
        flash_attn = self.config.get('flash_attn')
        if flash_attn:
            if isinstance(flash_attn, str):
                # Value-bearing form: --flash-attn auto
                cmd.extend(["--flash-attn", flash_attn])
            else:
                # Boolean flag form
                cmd.append("--flash-attn")
        if self.config.get('no_mmap'):
            cmd.append("--no-mmap")

        # Add any additional args from config
        if self.config.get('additional_args'):
            cmd.extend(self.config['additional_args'])

        # Log hardware configuration being used
        logger.info("Hardware Configuration:")
        logger.info(f"  GPU Layers: {self.config.get('n_gpu_layers', -1)}")
        logger.info(f"  Context Size: {effective_ctx_size} tokens")
        logger.info(f"  CPU Threads: {self.config.get('threads', 'auto')}")
        if self.config.get('numa_mode'):
            logger.info(f"  NUMA Mode: {self.config['numa_mode']}")
        if self.config.get('batch_size'):
            logger.info(f"  Batch Size: {self.config['batch_size']}")
        if self.config.get('ubatch_size'):
            logger.info(f"  UBatch Size: {self.config['ubatch_size']}")
        if self.config.get('mlock'):
            logger.info(f"  Memory Lock: enabled")
        if self.config.get('threads_batch'):
            logger.info(f"  Threads (batch/prefill): {self.config['threads_batch']}")
        if self.config.get('flash_attn'):
            logger.info(f"  Flash Attention: enabled")
        if self.config.get('no_mmap'):
            logger.info(f"  No MMap (preload): enabled")

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
                self.last_crash_info = None  # Clear any previous crash info on healthy start
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

        error_keywords = ('error', 'fatal', 'abort', 'exception', 'failed', 'segfault', 'signal')
        warn_keywords = ('warn', 'invalid', 'unsupported', 'cannot', 'not found')
        info_keywords = ('download', 'fetching', 'progress', 'mb', 'loaded', 'ready')

        for line in self.process.stderr:
            line = line.strip()
            if not line:
                continue
            line_lower = line.lower()
            if any(kw in line_lower for kw in error_keywords):
                logger.error(f"llama-server: {line}")
            elif any(kw in line_lower for kw in warn_keywords):
                logger.warning(f"llama-server: {line}")
            elif any(kw in line_lower for kw in info_keywords):
                logger.info(f"llama-server: {line}")
            else:
                logger.debug(f"llama-server: {line}")

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
    
    def _kill_process_tree(self, pid: int) -> None:
        """
        Kill a process and all its children using psutil.

        This prevents orphaned llama-server child processes from lingering
        after the parent is stopped.

        Args:
            pid: Process ID of the root process to kill
        """
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)

            # Terminate children first, then parent
            for child in children:
                try:
                    logger.debug(f"Terminating child process {child.pid} ({child.name()})")
                    child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            # Terminate parent
            try:
                parent.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

            # Wait for all to exit (parent + children)
            gone, alive = psutil.wait_procs(children + [parent], timeout=10)

            # Force kill any survivors
            for proc in alive:
                try:
                    logger.warning(f"Force killing process {proc.pid} ({proc.name()})")
                    proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            if alive:
                # Final wait after force kill
                psutil.wait_procs(alive, timeout=5)

        except psutil.NoSuchProcess:
            logger.debug(f"Process {pid} already exited")
        except Exception as e:
            logger.error(f"Error killing process tree for PID {pid}: {e}")

    def stop(self) -> bool:
        """
        Stop the llama-server process gracefully, killing the entire process tree.

        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.process:
            logger.warning("No llama-server process to stop")
            return True

        pid = self.process.pid
        logger.info(f"Stopping llama-server (PID: {pid})")

        try:
            # Use process tree killing for thorough cleanup
            self._kill_process_tree(pid)

            # Clean up the Popen object
            try:
                self.process.wait(timeout=2)
            except (subprocess.TimeoutExpired, OSError):
                pass

            self.is_running = False
            self.process = None
            logger.info("llama-server stopped successfully")
            return True

        except Exception as e:
            logger.error(f"Error stopping llama-server: {e}")
            # Last resort: try direct kill on the Popen object
            try:
                self.process.kill()
                self.process.wait(timeout=5)
            except Exception:
                pass
            self.is_running = False
            self.process = None
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
    
    def idle_unload(self) -> bool:
        """
        Stop llama-server due to idle timeout, preserving model info for reload.

        Unlike stop(), this sets the idle_unloaded flag so the server knows
        it can automatically reload when the next request arrives.

        Returns:
            True if unloaded successfully, False otherwise
        """
        if not self.is_running:
            return True

        logger.info("Idle timeout reached - unloading model to free resources")
        result = self.stop()
        if result:
            self.idle_unloaded = True
            logger.info(f"Model unloaded. Will reload '{self.last_model_identifier}' on next request.")
        return result

    def reload_after_idle(self) -> bool:
        """
        Reload the last model after an idle unload.

        Returns:
            True if reloaded successfully, False otherwise
        """
        if self.is_running:
            logger.warning("llama-server is already running, no reload needed")
            self.idle_unloaded = False
            return True

        if not self.last_model_identifier:
            logger.error("Cannot reload: no previous model information stored")
            self.idle_unloaded = False
            return False

        logger.info(f"Reloading model after idle unload: {self.last_model_identifier}")
        result = self.start(
            self.last_model_identifier,
            use_hf=self.last_use_hf,
            ctx_size=self.last_ctx_size
        )

        if result:
            self.idle_unloaded = False
            logger.info("Model reloaded successfully after idle unload")
        else:
            logger.error("Failed to reload model after idle unload")

        return result

    def is_healthy(self) -> bool:
        """
        Check if llama-server is running and responsive.

        When the process is found dead, captures exit code and remaining stderr
        into self.last_crash_info for surfacing via /health and logs.

        Returns:
            True if healthy, False otherwise
        """
        if not self.is_running or not self.process:
            return False

        # Check if process is still alive
        if self.process.poll() is not None:
            exit_code = self.process.returncode

            # Capture remaining stderr for diagnostics
            stderr_tail = ""
            try:
                remaining = self.process.stderr.read() if self.process.stderr else ""
                if remaining and remaining.strip():
                    # Keep last 500 chars to avoid huge dumps
                    stderr_tail = remaining.strip()[-500:]
            except Exception:
                pass

            self.last_crash_info = {
                "exit_code": exit_code,
                "stderr": stderr_tail,
                "timestamp": time.time(),
            }

            logger.error(
                f"llama-server process has terminated (exit code: {exit_code})"
            )
            if stderr_tail:
                logger.error(f"llama-server stderr: {stderr_tail}")

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
        self.idle_unloaded = False