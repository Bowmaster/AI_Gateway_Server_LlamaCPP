"""
hardware_detector.py - Hardware Detection and Configuration Optimization

Auto-detects GPU, CPU, and RAM capabilities and generates optimal configurations
for AI Lab server running on different hardware profiles.

Supports:
- NVIDIA GPU detection via pynvml (with nvidia-smi fallback)
- CPU detection via psutil
- RAM detection via psutil
- Optimal context window calculation based on available resources
- Configuration profile persistence to .hardware_profile.json
"""

import json
import logging
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import optional dependencies with graceful fallback
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logger.warning("psutil not installed - CPU/RAM detection will be limited")

try:
    # Note: Install nvidia-ml-py package (replaces deprecated pynvml)
    # The package is called nvidia-ml-py but imports as pynvml
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False
    logger.info("nvidia-ml-py not installed - will use nvidia-smi fallback for GPU detection")


# ============================================================================
# GPU Detection
# ============================================================================

def detect_nvidia_gpu() -> Dict[str, Any]:
    """
    Detect NVIDIA GPU using pynvml (preferred) or nvidia-smi (fallback).

    Returns:
        Dictionary with GPU information:
        {
            "has_gpu": bool,
            "name": str,
            "vram_gb": float,
            "cuda_version": str,
            "driver_version": str,
            "compute_capability": str (optional)
        }
    """
    # Try pynvml first (faster and more reliable)
    if HAS_PYNVML:
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            if device_count > 0:
                # Get first GPU (primary)
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                name = pynvml.nvmlDeviceGetName(handle)

                # Get VRAM in GB
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_gb = mem_info.total / (1024 ** 3)

                # Get driver version
                driver_version = pynvml.nvmlSystemGetDriverVersion()

                # Get CUDA version
                cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
                cuda_major = cuda_version // 1000
                cuda_minor = (cuda_version % 1000) // 10
                cuda_version_str = f"{cuda_major}.{cuda_minor}"

                # Get compute capability
                try:
                    major = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[0]
                    minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[1]
                    compute_capability = f"{major}.{minor}"
                except:
                    compute_capability = "unknown"

                pynvml.nvmlShutdown()

                logger.info(f"GPU detected via pynvml: {name} ({vram_gb:.1f}GB VRAM)")

                return {
                    "has_gpu": True,
                    "name": name.decode() if isinstance(name, bytes) else name,
                    "vram_gb": round(vram_gb, 2),
                    "cuda_version": cuda_version_str,
                    "driver_version": driver_version.decode() if isinstance(driver_version, bytes) else driver_version,
                    "compute_capability": compute_capability,
                    "detection_method": "pynvml"
                }
            else:
                logger.info("No NVIDIA GPU detected via pynvml")
                pynvml.nvmlShutdown()

        except Exception as e:
            logger.warning(f"pynvml detection failed: {e}, trying nvidia-smi fallback")

    # Fallback to nvidia-smi subprocess
    return _detect_nvidia_gpu_via_smi()


def _detect_nvidia_gpu_via_smi() -> Dict[str, Any]:
    """Detect NVIDIA GPU using nvidia-smi command-line tool (fallback method)."""
    try:
        # Run nvidia-smi to get GPU info
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0 and result.stdout.strip():
            # Parse first GPU
            line = result.stdout.strip().split('\n')[0]
            parts = [p.strip() for p in line.split(',')]

            if len(parts) >= 3:
                name = parts[0]
                vram_mb = float(parts[1])
                vram_gb = vram_mb / 1024
                driver_version = parts[2]

                # Try to get CUDA version
                cuda_version = "unknown"
                try:
                    cuda_result = subprocess.run(
                        ["nvidia-smi"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if cuda_result.returncode == 0:
                        # Parse CUDA version from header
                        for line in cuda_result.stdout.split('\n'):
                            if "CUDA Version:" in line:
                                cuda_version = line.split("CUDA Version:")[1].strip().split()[0]
                                break
                except:
                    pass

                logger.info(f"GPU detected via nvidia-smi: {name} ({vram_gb:.1f}GB VRAM)")

                return {
                    "has_gpu": True,
                    "name": name,
                    "vram_gb": round(vram_gb, 2),
                    "cuda_version": cuda_version,
                    "driver_version": driver_version,
                    "detection_method": "nvidia-smi"
                }

        logger.info("No NVIDIA GPU detected via nvidia-smi")

    except FileNotFoundError:
        logger.info("nvidia-smi not found - no NVIDIA GPU available")
    except Exception as e:
        logger.warning(f"nvidia-smi detection failed: {e}")

    # No GPU detected
    return {
        "has_gpu": False,
        "name": "None",
        "vram_gb": 0.0,
        "cuda_version": "N/A",
        "driver_version": "N/A",
        "detection_method": "none"
    }


# ============================================================================
# CPU Detection
# ============================================================================

def detect_cpu() -> Dict[str, Any]:
    """
    Detect CPU information using psutil.

    Returns:
        Dictionary with CPU information:
        {
            "name": str,
            "physical_cores": int,
            "logical_cores": int,
            "max_frequency_mhz": float (optional),
            "architecture": str
        }
    """
    if not HAS_PSUTIL:
        logger.warning("psutil not available - using basic CPU detection")
        return {
            "name": platform.processor() or "Unknown",
            "physical_cores": os.cpu_count() or 1,
            "logical_cores": os.cpu_count() or 1,
            "architecture": platform.machine(),
            "detection_method": "basic"
        }

    try:
        # Get CPU counts
        physical_cores = psutil.cpu_count(logical=False) or 1
        logical_cores = psutil.cpu_count(logical=True) or 1

        # Get CPU frequency (may not be available on all systems)
        max_freq = None
        try:
            freq = psutil.cpu_freq()
            if freq:
                max_freq = freq.max
        except:
            pass

        # Get CPU name (platform-specific)
        cpu_name = platform.processor()

        # On Windows, try to get more detailed name
        if platform.system() == "Windows":
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
                winreg.CloseKey(key)
            except:
                pass

        logger.info(f"CPU detected: {cpu_name} ({physical_cores} cores, {logical_cores} threads)")

        result = {
            "name": cpu_name or "Unknown",
            "physical_cores": physical_cores,
            "logical_cores": logical_cores,
            "architecture": platform.machine(),
            "detection_method": "psutil"
        }

        if max_freq:
            result["max_frequency_mhz"] = round(max_freq, 0)

        return result

    except Exception as e:
        logger.error(f"CPU detection failed: {e}")
        return {
            "name": "Unknown",
            "physical_cores": os.cpu_count() or 1,
            "logical_cores": os.cpu_count() or 1,
            "architecture": platform.machine(),
            "detection_method": "fallback"
        }


# ============================================================================
# Memory Detection
# ============================================================================

def detect_memory() -> Dict[str, Any]:
    """
    Detect system memory (RAM) using psutil.

    Returns:
        Dictionary with memory information:
        {
            "total_gb": float,
            "available_gb": float,
            "percent_available": float
        }
    """
    if not HAS_PSUTIL:
        logger.warning("psutil not available - cannot detect RAM")
        return {
            "total_gb": 0.0,
            "available_gb": 0.0,
            "percent_available": 0.0,
            "detection_method": "unavailable"
        }

    try:
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024 ** 3)
        available_gb = mem.available / (1024 ** 3)
        percent_available = (available_gb / total_gb) * 100

        logger.info(f"RAM detected: {total_gb:.1f}GB total ({available_gb:.1f}GB available)")

        return {
            "total_gb": round(total_gb, 2),
            "available_gb": round(available_gb, 2),
            "percent_available": round(percent_available, 1),
            "detection_method": "psutil"
        }

    except Exception as e:
        logger.error(f"Memory detection failed: {e}")
        return {
            "total_gb": 0.0,
            "available_gb": 0.0,
            "percent_available": 0.0,
            "detection_method": "error"
        }


# ============================================================================
# NUMA / Multi-Socket Detection
# ============================================================================

def detect_numa_topology(cpu_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect NUMA topology (multi-socket systems) for optimal thread/memory placement.

    Detection strategy (in priority order):
    1. Windows: wmic cpu get SocketDesignation to count physical sockets
    2. Linux: count directories in /sys/devices/system/node/
    3. Heuristic fallback: high physical core count + server CPU name keywords

    Args:
        cpu_info: Dictionary from detect_cpu()

    Returns:
        Dictionary with NUMA information:
        {
            "numa_nodes": int,
            "is_multi_socket": bool,
            "detection_method": str
        }
    """
    system = platform.system()

    # Windows: query WMI for socket count
    if system == "Windows":
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "SocketDesignation"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                # Each non-empty, non-header line is a socket
                lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
                # First line is the header "SocketDesignation"
                sockets = max(1, len(lines) - 1)
                logger.info(f"NUMA detection (wmic): {sockets} socket(s)")
                return {
                    "numa_nodes": sockets,
                    "is_multi_socket": sockets > 1,
                    "detection_method": "wmic"
                }
        except Exception as e:
            logger.debug(f"wmic NUMA detection failed: {e}")

    # Linux: count NUMA node directories
    if system == "Linux":
        try:
            numa_path = Path("/sys/devices/system/node")
            if numa_path.exists():
                nodes = [d for d in numa_path.iterdir() if d.is_dir() and d.name.startswith("node")]
                node_count = max(1, len(nodes))
                logger.info(f"NUMA detection (sysfs): {node_count} node(s)")
                return {
                    "numa_nodes": node_count,
                    "is_multi_socket": node_count > 1,
                    "detection_method": "sysfs"
                }
        except Exception as e:
            logger.debug(f"sysfs NUMA detection failed: {e}")

    # Heuristic fallback: server CPUs with high core counts are likely multi-socket
    physical_cores = cpu_info.get("physical_cores", 0)
    cpu_name = cpu_info.get("name", "").lower()
    server_keywords = ("xeon", "epyc", "e5-", "e7-", "e3-", "platinum", "gold", "silver")

    is_server_cpu = any(kw in cpu_name for kw in server_keywords)
    if is_server_cpu and physical_cores >= 20:
        # High core-count server CPU — likely dual-socket
        inferred_sockets = 2
        logger.info(f"NUMA detection (heuristic): inferred {inferred_sockets} sockets "
                     f"({physical_cores} cores, server CPU '{cpu_info.get('name', '')}')")
        return {
            "numa_nodes": inferred_sockets,
            "is_multi_socket": True,
            "detection_method": "heuristic"
        }

    # Single socket assumed
    logger.info("NUMA detection: single socket (default)")
    return {
        "numa_nodes": 1,
        "is_multi_socket": False,
        "detection_method": "default"
    }


# ============================================================================
# Configuration Generation
# ============================================================================

def calculate_optimal_context(
    vram_gb: Optional[float] = None,
    ram_gb: Optional[float] = None,
    model_size_gb: float = 5.0,
    mode: str = "auto"
) -> int:
    """
    Calculate optimal context window size based on available resources.

    Args:
        vram_gb: GPU VRAM in GB (if GPU mode)
        ram_gb: System RAM in GB (if CPU mode)
        model_size_gb: Estimated model size in GB (default 5GB for 7B Q4)
        mode: "gpu", "cpu", or "auto"

    Returns:
        Optimal context size in tokens
    """
    # Constants for estimation
    # Note: KV cache size varies by model architecture and precision
    # For typical 7B models: ~2-4 bytes per token
    # We use 3 bytes as a conservative middle ground
    TOKEN_SIZE_BYTES = 3.0  # Conservative estimate for KV cache per token
    GB_TO_BYTES = 1024 ** 3
    SAFETY_BUFFER_GB = 2.0  # Reserve for system/overhead

    if mode == "gpu" or (mode == "auto" and vram_gb and vram_gb > 0):
        # GPU mode - use empirical data for practical context sizes
        # Based on real-world llama.cpp usage with 7B Q4 models
        if not vram_gb or vram_gb < 4:
            return 8192  # Minimal context for small GPU

        # Use empirical VRAM-to-context mapping (conservative, tested values)
        # These account for model size + KV cache + overhead
        # Capped at 32K to match most model context limits (Qwen, Mistral, DeepSeek)
        if vram_gb >= 16:
            return 32768  # 32K context - safe for most models
        elif vram_gb >= 12:
            return 24576  # 24K context
        elif vram_gb >= 8:
            return 16384  # 16K context
        elif vram_gb >= 6:
            return 12288  # 12K context
        else:
            return 8192   # 8K context for 4-6GB VRAM

    elif mode == "cpu" or (mode == "auto" and ram_gb):
        # CPU mode: conservative context optimizes for speed over capacity.
        # Large KV caches destroy memory bandwidth on CPU, causing severe
        # tok/s degradation even when RAM is abundant. 8K is sufficient for
        # interactive use. Users can override via CTX_SIZE env var when they
        # accept the speed tradeoff.
        if not ram_gb or ram_gb < 8:
            return 4096  # Minimal for low RAM
        elif ram_gb >= 16:
            return 8192  # Speed-optimized default for 16GB+
        else:
            return 4096

    # Fallback
    return 8192


def generate_optimal_config(hardware_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate optimal llama-server configuration based on detected hardware.

    Args:
        hardware_info: Dictionary with gpu, cpu, and memory info

    Returns:
        Dictionary with recommended config:
        {
            "n_gpu_layers": int,
            "ctx_size": int,
            "threads": int or None,
            "mode": str,
            "reasoning": str
        }
    """
    gpu = hardware_info.get("gpu", {})
    cpu = hardware_info.get("cpu", {})
    memory = hardware_info.get("memory", {})

    has_gpu = gpu.get("has_gpu", False)
    vram_gb = gpu.get("vram_gb", 0.0)
    logical_cores = cpu.get("logical_cores", 1)
    ram_gb = memory.get("total_gb", 0.0)

    # Determine system type and optimal configuration
    if has_gpu and vram_gb >= 8:
        # GPU-primary mode: Sufficient VRAM for full GPU offloading
        ctx_size = calculate_optimal_context(vram_gb=vram_gb, mode="gpu")

        return {
            "n_gpu_layers": -1,  # All layers on GPU
            "ctx_size": ctx_size,
            "threads": None,  # GPU doesn't need thread config
            "mode": "gpu_primary",
            "reasoning": f"GPU with {vram_gb}GB VRAM detected - using full GPU acceleration"
        }

    elif has_gpu and 4 <= vram_gb < 8:
        # GPU-partial mode: Some VRAM, but limited
        # Offload most layers to GPU, keep some on CPU for large models
        ctx_size = calculate_optimal_context(vram_gb=vram_gb, mode="gpu")

        return {
            "n_gpu_layers": -1,  # Try full GPU, may need manual tuning for large models
            "ctx_size": ctx_size,
            "threads": None,
            "mode": "gpu_limited",
            "reasoning": f"GPU with {vram_gb}GB VRAM detected - may need hybrid mode for large models"
        }

    elif has_gpu and vram_gb < 4 and ram_gb > 64:
        # Hybrid mode: Small GPU but lots of RAM
        # Offload some layers to GPU, use CPU for context
        ctx_size = calculate_optimal_context(ram_gb=ram_gb, mode="cpu")
        threads = max(1, logical_cores - 4)  # Reserve some threads

        return {
            "n_gpu_layers": 10,  # Partial GPU offloading
            "ctx_size": ctx_size,
            "threads": threads,
            "mode": "hybrid_ram_optimized",
            "reasoning": f"Small GPU ({vram_gb}GB) + high RAM ({ram_gb}GB) - hybrid mode with large context"
        }

    elif not has_gpu and ram_gb >= 128:
        # CPU high-RAM mode: No GPU but massive RAM (like System 2)
        ctx_size = calculate_optimal_context(ram_gb=ram_gb, mode="cpu")
        physical_cores = cpu.get("physical_cores", logical_cores)
        threads = max(1, physical_cores - 2)  # Physical cores only, avoids HT contention

        # Read NUMA topology from hardware_info (populated by detect_and_save)
        numa = hardware_info.get("numa", {})
        is_multi_socket = numa.get("is_multi_socket", False)

        return {
            "n_gpu_layers": 0,  # All on CPU
            "ctx_size": ctx_size,
            "threads": threads,
            "mode": "cpu_highram",
            "reasoning": (f"No GPU, {ram_gb}GB RAM, {physical_cores} physical cores "
                          f"({logical_cores} logical) - CPU inference with physical-core threading"
                          f"{', NUMA distribute' if is_multi_socket else ''}"),
            "cpu_optimization": {
                "numa_mode": "distribute" if is_multi_socket else None,
                "batch_size": 512,
                "ubatch_size": 512,
                "mlock": True,
                # Prefill uses all logical cores (HT helps for parallel prompt eval)
                "threads_batch": logical_cores,
                "flash_attn": True,
                # Preload entire model into RAM (avoids mmap page faults, safe with 128GB+)
                "no_mmap": True,
            }
        }

    elif not has_gpu and ram_gb < 128:
        # CPU standard mode: No GPU, normal RAM
        ctx_size = calculate_optimal_context(ram_gb=ram_gb, mode="cpu")
        physical_cores = cpu.get("physical_cores", logical_cores)
        threads = max(1, physical_cores - 2)  # Physical cores only, avoids HT contention

        return {
            "n_gpu_layers": 0,  # All on CPU
            "ctx_size": ctx_size,
            "threads": threads,
            "mode": "cpu_standard",
            "reasoning": (f"No GPU, {ram_gb}GB RAM, {physical_cores} physical cores "
                          f"({logical_cores} logical) - CPU-only mode"),
            "cpu_optimization": {
                "numa_mode": None,
                "batch_size": 512,
                "ubatch_size": 512,
                "mlock": ram_gb >= 32,
                "threads_batch": logical_cores,
                "flash_attn": True,
                "no_mmap": False,  # Don't preload on lower-RAM systems
            }
        }

    else:
        # Fallback to conservative defaults
        return {
            "n_gpu_layers": 0,
            "ctx_size": 8192,
            "threads": max(1, logical_cores - 2) if logical_cores > 2 else None,
            "mode": "fallback",
            "reasoning": "Using conservative defaults due to uncertain hardware detection"
        }


def classify_system_type(hardware_info: Dict[str, Any]) -> str:
    """
    Classify system into a type based on hardware profile.

    Returns:
        One of: "gaming_pc", "workstation", "server", "laptop", "basic"
    """
    gpu = hardware_info.get("gpu", {})
    cpu = hardware_info.get("cpu", {})
    memory = hardware_info.get("memory", {})

    has_gpu = gpu.get("has_gpu", False)
    vram_gb = gpu.get("vram_gb", 0.0)
    logical_cores = cpu.get("logical_cores", 1)
    ram_gb = memory.get("total_gb", 0.0)

    # Server: High core count, high RAM, typically no GPU
    if logical_cores >= 32 and ram_gb >= 128:
        return "server"

    # Workstation: High-end GPU or high core count with decent RAM
    if (vram_gb >= 16 and ram_gb >= 64) or (logical_cores >= 16 and ram_gb >= 64):
        return "workstation"

    # Gaming PC: Good GPU with moderate CPU/RAM
    if has_gpu and vram_gb >= 8 and logical_cores >= 8 and ram_gb >= 16:
        return "gaming_pc"

    # Laptop: Lower specs, integrated or mobile GPU
    if logical_cores <= 8 and ram_gb <= 32:
        return "laptop"

    return "basic"


# ============================================================================
# Profile Persistence
# ============================================================================

def save_hardware_profile(profile: Dict[str, Any], path: str = ".hardware_profile.json") -> bool:
    """
    Save hardware profile to JSON file.

    Args:
        profile: Hardware profile dictionary
        path: Path to save file (default: .hardware_profile.json)

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(path, 'w') as f:
            json.dump(profile, f, indent=2)

        logger.info(f"Hardware profile saved to {path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save hardware profile: {e}")
        return False


def load_hardware_profile(path: str = ".hardware_profile.json") -> Optional[Dict[str, Any]]:
    """
    Load hardware profile from JSON file.

    Args:
        path: Path to profile file (default: .hardware_profile.json)

    Returns:
        Profile dictionary if found, None otherwise
    """
    if not os.path.exists(path):
        logger.info(f"No hardware profile found at {path}")
        return None

    try:
        with open(path, 'r') as f:
            profile = json.load(f)

        logger.info(f"Hardware profile loaded from {path}")
        return profile

    except Exception as e:
        logger.error(f"Failed to load hardware profile: {e}")
        return None


# ============================================================================
# Main Detection Entry Point
# ============================================================================

def detect_and_save(path: str = ".hardware_profile.json") -> Dict[str, Any]:
    """
    Detect hardware, generate optimal config, and save profile.

    This is the main entry point for hardware detection.

    Args:
        path: Path to save profile (default: .hardware_profile.json)

    Returns:
        Complete hardware profile dictionary
    """
    logger.info("Starting hardware detection...")

    # Detect all hardware components
    gpu_info = detect_nvidia_gpu()
    cpu_info = detect_cpu()
    memory_info = detect_memory()
    numa_info = detect_numa_topology(cpu_info)

    # Combine into hardware info
    hardware_info = {
        "gpu": gpu_info,
        "cpu": cpu_info,
        "memory": memory_info,
        "numa": numa_info
    }

    # Generate optimal configuration
    recommended_config = generate_optimal_config(hardware_info)

    # Classify system type
    system_type = classify_system_type(hardware_info)

    # Build complete profile
    profile = {
        "version": "1.1",
        "detected_at": datetime.now().isoformat(),
        "system_type": system_type,
        "gpu": gpu_info,
        "cpu": cpu_info,
        "memory": memory_info,
        "numa": numa_info,
        "recommended_config": recommended_config,
        "manual_overrides": {}  # User can add custom overrides here
    }

    # Save to file
    save_hardware_profile(profile, path)

    # Log summary
    logger.info("=" * 60)
    logger.info("Hardware Detection Summary")
    logger.info("=" * 60)
    logger.info(f"System Type: {system_type}")
    if gpu_info["has_gpu"]:
        logger.info(f"GPU: {gpu_info['name']} ({gpu_info['vram_gb']}GB VRAM)")
    else:
        logger.info("GPU: None")
    logger.info(f"CPU: {cpu_info['name']} ({cpu_info['logical_cores']} threads, "
                f"{cpu_info.get('physical_cores', '?')} physical cores)")
    logger.info(f"NUMA: {numa_info['numa_nodes']} node(s) "
                f"({'multi-socket' if numa_info['is_multi_socket'] else 'single-socket'}, "
                f"via {numa_info['detection_method']})")
    logger.info(f"RAM: {memory_info['total_gb']:.1f}GB")
    logger.info(f"Recommended Mode: {recommended_config['mode']}")
    logger.info(f"Context Size: {recommended_config['ctx_size']} tokens")
    logger.info(f"Threads: {recommended_config.get('threads', 'auto')}")
    cpu_opt = recommended_config.get('cpu_optimization')
    if cpu_opt:
        logger.info(f"CPU Optimization: NUMA={cpu_opt.get('numa_mode', 'none')}, "
                     f"batch={cpu_opt.get('batch_size')}, "
                     f"ubatch={cpu_opt.get('ubatch_size')}, "
                     f"mlock={cpu_opt.get('mlock')}")
    logger.info(f"Reasoning: {recommended_config['reasoning']}")
    logger.info("=" * 60)

    return profile


# ============================================================================
# CLI Entry Point (for testing)
# ============================================================================

if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run detection
    profile = detect_and_save()

    # Pretty print results
    print("\n" + "=" * 60)
    print("Hardware Profile Generated")
    print("=" * 60)
    print(json.dumps(profile, indent=2))
    print("=" * 60)
