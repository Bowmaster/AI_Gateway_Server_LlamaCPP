"""
tools.py - Native tools for AI Lab Server
"""

import socket
import subprocess
import platform
import time
import os
from typing import Dict, Optional, List, Callable
from pathlib import Path

# =============================================================================
# TOOL DECORATOR SYSTEM
# =============================================================================

_TOOL_REGISTRY: Dict[str, Dict] = {}  # Private registry for tool safety


def tool(name: str, description: str, parameters: dict, key_param: Optional[str] = None):
    """
    Decorator to register a function as a callable tool.

    Security: Only explicitly decorated functions can be called as tools.
    No dynamic registration or injection is possible.

    Args:
        name: Tool name for API
        description: What the tool does (shown to LLM)
        parameters: OpenAI-format parameter schema
        key_param: Primary parameter for logging (e.g., 'hostname', 'path')

    Example:
        @tool(
            name="lookup_hostname",
            description="Look up IP address...",
            parameters={...},
            key_param="hostname"
        )
        def lookup_hostname(hostname: str) -> dict:
            ...
    """
    def decorator(func: Callable) -> Callable:
        _TOOL_REGISTRY[name] = {
            'function': func,
            'key_param': key_param,
            'definition': {
                'name': name,
                'description': description,
                'parameters': parameters
            }
        }
        return func
    return decorator


def get_available_tools() -> List[dict]:
    """
    Return list of tool definitions for LLM.

    Returns:
        List of tool definitions in OpenAI format
    """
    return [entry['definition'] for entry in _TOOL_REGISTRY.values()]


def execute_tool(name: str, arguments: dict) -> dict:
    """
    Execute a registered tool by name.

    Security: Only calls functions in _TOOL_REGISTRY (whitelist).

    Args:
        name: Tool name to execute
        arguments: Dict of arguments to pass to tool

    Returns:
        Tool result dictionary
    """
    if name not in _TOOL_REGISTRY:
        return {"error": f"Unknown tool: {name}"}

    try:
        func = _TOOL_REGISTRY[name]['function']
        return func(**arguments)
    except Exception as e:
        return {"error": f"Tool execution failed: {str(e)}"}


def get_tool_key_param(name: str) -> str:
    """
    Get the key parameter name for a tool (for logging).

    Args:
        name: Tool name

    Returns:
        Key parameter name or 'unknown' if not found
    """
    return _TOOL_REGISTRY.get(name, {}).get('key_param', 'unknown')


# =============================================================================
# NETWORKING TOOLS
# =============================================================================

@tool(
    name="lookup_hostname",
    description="Look up the IP address of a hostname/domain and check if it's reachable. Use this when asked about current IPs, domain resolution, or network connectivity.",
    parameters={
        "type": "object",
        "properties": {
            "hostname": {
                "type": "string",
                "description": "The hostname or domain name to look up (e.g., 'google.com', 'github.com')"
            }
        },
        "required": ["hostname"]
    },
    key_param="hostname"
)
def lookup_hostname(hostname: str) -> Dict[str, any]:
    """
    Resolve hostname to IP address and optionally ping it.

    Args:
        hostname: Domain name or hostname to look up

    Returns:
        Dict with ip_address, hostname, and ping_result
    """
    result = {
        "hostname": hostname,
        "ip_address": None,
        "ping_success": None,
        "ping_time_ms": None,
        "error": None
    }
    
    try:
        # DNS lookup
        ip_address = socket.gethostbyname(hostname)
        result["ip_address"] = ip_address
        
        # Try to ping (quick check)
        param = "-n" if platform.system().lower() == "windows" else "-c"
        command = ["ping", param, "1", "-w" if platform.system().lower() == "windows" else "-W", "2", hostname]
        
        ping_result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3,
            text=True
        )
        
        result["ping_success"] = ping_result.returncode == 0
        
        # Extract ping time if successful
        if result["ping_success"]:
            output = ping_result.stdout
            # Parse ping time (rough extraction, works for most cases)
            if "time=" in output:
                time_str = output.split("time=")[1].split("ms")[0].strip()
                try:
                    result["ping_time_ms"] = float(time_str)
                except:
                    pass
        
        return result
        
    except socket.gaierror:
        result["error"] = f"Could not resolve hostname: {hostname}"
        return result
    except subprocess.TimeoutExpired:
        result["error"] = "Ping timed out"
        return result
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
        return result

@tool(
    name="measure_http_latency",
    description="Determine the latency time required to access a domain name via the HTTPS protocol",
    parameters={
        "type": "object",
        "properties": {
            "hostname": {
                "type": "string",
                "description": "The hostname or domain name to test latency for (e.g., 'google.com', 'github.com')"
            }
        },
        "required": ["hostname"]
    },
    key_param="hostname"
)
def measure_http_latency(hostname: str) -> dict:
    """Measure actual HTTP response time"""
    import requests
    try:
        start = time.time()
        requests.get(f"https://{hostname}", timeout=5)
        latency = (time.time() - start) * 1000
        return {"hostname": hostname, "http_latency_ms": latency}
    except Exception as e:
        return {"hostname": hostname, "http_latency_ms": None, "error": str(e)}

# =============================================================================
# FILE SYSTEM TOOLS
# =============================================================================

def _is_protected_path(path: str) -> bool:
    """
    Check if a path is a protected system directory.
    
    Protected paths:
    - Windows: C:\Windows, C:\Program Files, C:\Program Files (x86), C:\
    - Linux: /, /etc, /proc, /sbin, /boot, /sys, /dev
    """
    # Normalize path
    path = os.path.abspath(path)
    
    system_protected = {
        "Windows": [
            "C:\\Windows",
            "C:\\Program Files",
            "C:\\Program Files (x86)",
            "C:\\"  # Root drive
        ],
        "Linux": [
            "/",  # Root
            "/etc",
            "/proc",
            "/sbin",
            "/boot",
            "/sys",
            "/dev"
        ]
    }
    
    os_name = platform.system()
    protected_dirs = system_protected.get(os_name, [])
    
    # Check if path is exactly a protected directory or starts with one
    for protected in protected_dirs:
        # Normalize protected path for comparison
        protected_norm = os.path.abspath(protected)
        
        # Exact match (e.g., listing C:\ or /)
        if path == protected_norm:
            return True
        
        # Starts with protected path (e.g., C:\Windows\System32)
        # But allow subdirectories that aren't the root protected ones
        if os_name == "Windows":
            if path.startswith(protected_norm + "\\"):
                return True
        else:  # Linux/Unix
            if protected != "/" and path.startswith(protected_norm + "/"):
                return True
    
    return False

def _normalize_path(path: str) -> str:
    """
    Normalize and resolve a file path to an absolute path.

    Features:
    - Converts relative paths to absolute using current working directory
    - Handles mixed path separators (/ and \\)
    - Expands user home directory (~)
    - Normalizes path for the current OS

    Args:
        path: Relative or absolute file path

    Returns:
        Absolute normalized path for current OS
    """
    # Expand user home directory if present (~)
    path = os.path.expanduser(path)

    # If path is already absolute, just normalize it
    if os.path.isabs(path):
        return os.path.normpath(path)

    # Otherwise, resolve relative to current working directory
    cwd = os.getcwd()
    absolute_path = os.path.join(cwd, path)
    return os.path.normpath(absolute_path)

@tool(
    name="read_file",
    description="Read and return the contents of a text file. Supports both absolute and relative paths. Relative paths are resolved to the current working directory. Use this to examine code, configuration files, logs, or any text-based files.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to read from. Can be absolute (e.g., 'C:\\Users\\user\\file.txt') or relative (e.g., 'file.txt', './data/file.txt'). Relative paths are resolved to the current working directory."
            }
        },
        "required": ["path"]
    },
    key_param="path"
)
def read_file(path: str) -> dict:
    """
    Read and return the contents of a file.

    Args:
        path: The file path to read from (absolute or relative)

    Returns:
        Dict with path, lines (content), and optional error
    """
    try:
        # Normalize path to absolute
        path = _normalize_path(path)

        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
        
        return {
            "path": path,
            "lines": content,
            "size_bytes": len(content.encode("utf-8"))
        }
    except FileNotFoundError:
        return {"path": path, "lines": None, "error": "File not found"}
    except PermissionError:
        return {"path": path, "lines": None, "error": "Permission denied"}
    except UnicodeDecodeError:
        return {"path": path, "lines": None, "error": "File is not a text file (binary content)"}
    except Exception as e:
        return {"path": path, "lines": None, "error": str(e)}

@tool(
    name="write_file",
    description="Create or overwrite a file with the supplied content. Supports both absolute and relative paths. Relative paths are resolved to the current working directory. Safeguards prevent writing to system directories. Use this to create new files or update existing ones.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to write to. Can be absolute (e.g., 'C:\\Users\\user\\output.txt') or relative (e.g., 'output.txt', './data/output.txt'). Relative paths are resolved to the current working directory."
            },
            "lines": {
                "type": "string",
                "description": "The content to write to the file (as a string)"
            }
        },
        "required": ["path", "lines"]
    },
    key_param="path"
)
def write_file(path: str, lines: str) -> dict:
    """
    Write content to a file at the specified path.

    Args:
        path: The file path to write to (absolute or relative)
        lines: The content to write (string or list of strings)

    Safeguards:
        1. Relative paths are resolved to current working directory
        2. Parent directory must exist
        3. Cannot write to core system directories

    Returns:
        Dict with status and optional error
    """
    try:
        # Normalize path to absolute
        path = _normalize_path(path)

        # Extract the parent directory
        parent_dir = os.path.dirname(path)
        
        # Check if the parent directory exists
        if not os.path.exists(parent_dir):
            return {
                "path": path,
                "success": False,
                "error": f"Parent directory does not exist: {parent_dir}"
            }
        
        # Check if writing to a protected system directory
        if _is_protected_path(parent_dir):
            return {
                "path": path,
                "success": False,
                "error": f"Cannot write to protected system directory: {parent_dir}"
            }
        
        # Write the content
        with open(path, "w", encoding="utf-8") as file:
            if isinstance(lines, list):
                file.write("\n".join(lines))
            else:
                file.write(lines)
        
        return {
            "path": path,
            "success": True,
            "bytes_written": os.path.getsize(path)
        }
        
    except PermissionError:
        return {"path": path, "success": False, "error": "Permission denied"}
    except Exception as e:
        return {"path": path, "success": False, "error": str(e)}

@tool(
    name="list_contents",
    description="List all files and directories in a specified directory. Supports both absolute and relative paths. Returns files with sizes and directories. Protected system directories cannot be listed.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list contents of. Can be absolute or relative (e.g., 'C:\\Users\\user\\Documents', './data', or '.')"
            },
            "show_hidden": {
                "type": "boolean",
                "description": "Whether to include hidden files and directories (default: false)"
            }
        },
        "required": ["path"]
    },
    key_param="path"
)
def list_contents(path: str, show_hidden: bool = False) -> dict:
    """
    List files and directories in the specified path.

    Args:
        path: Directory path to list contents of (absolute or relative)
        show_hidden: Whether to show hidden files (default: False)

    Safeguards:
        - Cannot list root directories (C:\ on Windows, / on Linux)
        - Cannot list protected system directories

    Returns:
        Dict with files, directories, and optional error
    """
    try:
        # Normalize path to absolute
        path = _normalize_path(path)
        
        # Check if path exists
        if not os.path.exists(path):
            return {
                "path": path,
                "exists": False,
                "error": "Path does not exist"
            }
        
        # Check if it's a directory
        if not os.path.isdir(path):
            return {
                "path": path,
                "is_directory": False,
                "error": "Path is not a directory"
            }
        
        # Check if trying to list a protected directory
        if _is_protected_path(path):
            return {
                "path": path,
                "protected": True,
                "error": "Cannot list protected system directory"
            }
        
        # List contents
        entries = os.listdir(path)
        
        files = []
        directories = []
        
        for entry in entries:
            # Skip hidden files unless requested
            if not show_hidden and entry.startswith('.'):
                continue
            
            full_path = os.path.join(path, entry)
            
            try:
                if os.path.isfile(full_path):
                    size = os.path.getsize(full_path)
                    files.append({
                        "name": entry,
                        "size_bytes": size,
                        "size_human": _format_size(size)
                    })
                elif os.path.isdir(full_path):
                    directories.append({"name": entry})
            except (PermissionError, OSError):
                # Skip entries we can't access
                continue
        
        # Sort for consistent output
        files.sort(key=lambda x: x["name"].lower())
        directories.sort(key=lambda x: x["name"].lower())
        
        return {
            "path": path,
            "directories": directories,
            "files": files,
            "total_files": len(files),
            "total_directories": len(directories)
        }
        
    except PermissionError:
        return {"path": path, "error": "Permission denied"}
    except Exception as e:
        return {"path": path, "error": str(e)}

def _format_size(bytes: int) -> str:
    """Convert bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} PB"

@tool(
    name="search_files",
    description="Search for files matching a pattern within a directory. Supports both absolute and relative paths. Supports wildcards like *.py, test*.txt. Useful for finding specific files.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to search in. Can be absolute or relative (e.g., 'C:\\projects', './src', or '.')"
            },
            "pattern": {
                "type": "string",
                "description": "File pattern to match (e.g., '*.py', 'test*.txt', 'README.*')"
            },
            "recursive": {
                "type": "boolean",
                "description": "Search subdirectories recursively (default: true)"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 100)"
            }
        },
        "required": ["path", "pattern"]
    },
    key_param="pattern"
)
def search_files(path: str, pattern: str, recursive: bool = True, max_results: int = 100) -> dict:
    """
    Search for files matching a pattern in the specified directory.

    Args:
        path: Directory to search in (absolute or relative)
        pattern: Filename pattern (supports wildcards like *.py, test*.txt)
        recursive: Search subdirectories (default: True)
        max_results: Maximum number of results to return (default: 100)

    Safeguards:
        - Cannot search protected system directories
        - Limited to max_results to prevent overwhelming output

    Returns:
        Dict with matching files and their paths
    """
    import fnmatch

    try:
        # Normalize path to absolute
        path = _normalize_path(path)
        
        # Check if path exists
        if not os.path.exists(path):
            return {"path": path, "exists": False, "error": "Path does not exist"}
        
        # Check if it's a directory
        if not os.path.isdir(path):
            return {"path": path, "is_directory": False, "error": "Path is not a directory"}
        
        # Check if protected
        if _is_protected_path(path):
            return {"path": path, "protected": True, "error": "Cannot search protected system directory"}
        
        matches = []
        
        if recursive:
            # Recursive search using os.walk
            for root, dirs, files in os.walk(path):
                # Skip protected subdirectories
                dirs[:] = [d for d in dirs if not _is_protected_path(os.path.join(root, d))]
                
                for filename in files:
                    if fnmatch.fnmatch(filename, pattern):
                        full_path = os.path.join(root, filename)
                        try:
                            size = os.path.getsize(full_path)
                            matches.append({
                                "name": filename,
                                "path": full_path,
                                "size_bytes": size,
                                "size_human": _format_size(size)
                            })
                            
                            if len(matches) >= max_results:
                                break
                        except (PermissionError, OSError):
                            continue
                
                if len(matches) >= max_results:
                    break
        else:
            # Non-recursive search
            try:
                for filename in os.listdir(path):
                    if fnmatch.fnmatch(filename, pattern):
                        full_path = os.path.join(path, filename)
                        if os.path.isfile(full_path):
                            try:
                                size = os.path.getsize(full_path)
                                matches.append({
                                    "name": filename,
                                    "path": full_path,
                                    "size_bytes": size,
                                    "size_human": _format_size(size)
                                })
                                
                                if len(matches) >= max_results:
                                    break
                            except (PermissionError, OSError):
                                continue
            except PermissionError:
                return {"path": path, "error": "Permission denied"}
        
        return {
            "path": path,
            "pattern": pattern,
            "matches": matches,
            "total_found": len(matches),
            "truncated": len(matches) >= max_results
        }
        
    except Exception as e:
        return {"path": path, "pattern": pattern, "error": str(e)}

@tool(
    name="get_file_info",
    description="Get metadata about a file or directory without reading its contents. Supports both absolute and relative paths. Returns size, modification date, type, etc.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file or directory. Can be absolute or relative (e.g., 'file.txt', './data/file.txt')"
            }
        },
        "required": ["path"]
    },
    key_param="path"
)
def get_file_info(path: str) -> dict:
    """
    Get metadata about a file or directory without reading its contents.

    Args:
        path: Path to the file or directory (absolute or relative)

    Returns:
        Dict with size, modification time, type, permissions
    """
    try:
        # Normalize path to absolute
        path = _normalize_path(path)
        
        # Check if exists
        if not os.path.exists(path):
            return {"path": path, "exists": False, "error": "Path does not exist"}
        
        stat_info = os.stat(path)
        
        info = {
            "path": path,
            "exists": True,
            "is_file": os.path.isfile(path),
            "is_directory": os.path.isdir(path),
            "size_bytes": stat_info.st_size,
            "size_human": _format_size(stat_info.st_size),
            "modified_timestamp": stat_info.st_mtime,
            "modified_date": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat_info.st_mtime)),
            "created_timestamp": stat_info.st_ctime,
            "created_date": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat_info.st_ctime))
        }
        
        # Add file extension if it's a file
        if info["is_file"]:
            info["extension"] = os.path.splitext(path)[1]
        
        return info
        
    except PermissionError:
        return {"path": path, "error": "Permission denied"}
    except Exception as e:
        return {"path": path, "error": str(e)}

@tool(
    name="create_directory",
    description="Create a new directory. Supports both absolute and relative paths. Can create parent directories if needed. Cannot create in protected system locations.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to create. Can be absolute or relative (e.g., 'C:\\data', './new_folder', 'output')"
            },
            "parents": {
                "type": "boolean",
                "description": "Create parent directories if they don't exist (default: true)"
            }
        },
        "required": ["path"]
    },
    key_param="path"
)
def create_directory(path: str, parents: bool = True) -> dict:
    """
    Create a new directory.

    Args:
        path: Directory path to create (absolute or relative)
        parents: Create parent directories if needed (default: True)

    Safeguards:
        - Cannot create directories in protected system locations
        - Relative paths resolved to current working directory

    Returns:
        Dict with success status
    """
    try:
        # Normalize path to absolute
        path = _normalize_path(path)
        
        # Check if already exists
        if os.path.exists(path):
            if os.path.isdir(path):
                return {"path": path, "success": True, "already_exists": True}
            else:
                return {"path": path, "success": False, "error": "Path exists but is not a directory"}
        
        # Check parent directory protection
        parent = os.path.dirname(path)
        if _is_protected_path(parent):
            return {"path": path, "success": False, "error": f"Cannot create directory in protected location: {parent}"}
        
        # Create directory
        if parents:
            os.makedirs(path, exist_ok=True)
        else:
            os.mkdir(path)
        
        return {"path": path, "success": True, "created": True}
        
    except PermissionError:
        return {"path": path, "success": False, "error": "Permission denied"}
    except Exception as e:
        return {"path": path, "success": False, "error": str(e)}

@tool(
    name="move_file",
    description="Move or rename a file or directory. Supports both absolute and relative paths. Can be used to reorganize files or rename them.",
    parameters={
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Source path. Can be absolute or relative (e.g., 'old.txt', './data/file.txt')"
            },
            "destination": {
                "type": "string",
                "description": "Destination path. Can be absolute or relative (e.g., 'new.txt', './backup/file.txt')"
            },
            "overwrite": {
                "type": "boolean",
                "description": "Overwrite destination if it exists (default: false)"
            }
        },
        "required": ["source", "destination"]
    },
    key_param="source"
)
def move_file(source: str, destination: str, overwrite: bool = False) -> dict:
    """
    Move or rename a file or directory.

    Args:
        source: Source path (absolute or relative)
        destination: Destination path (absolute or relative)
        overwrite: Whether to overwrite if destination exists (default: False)

    Safeguards:
        - Cannot move from/to protected system directories
        - Relative paths resolved to current working directory
        - Won't overwrite unless explicitly allowed

    Returns:
        Dict with success status
    """
    import shutil

    try:
        # Normalize paths to absolute
        source = _normalize_path(source)
        destination = _normalize_path(destination)
        
        # Check if source exists
        if not os.path.exists(source):
            return {"source": source, "destination": destination, "success": False, "error": "Source does not exist"}
        
        # Check protection on source
        if _is_protected_path(source) or _is_protected_path(os.path.dirname(source)):
            return {"source": source, "destination": destination, "success": False, "error": "Cannot move from protected location"}
        
        # Check protection on destination
        dest_dir = os.path.dirname(destination)
        if _is_protected_path(destination) or _is_protected_path(dest_dir):
            return {"source": source, "destination": destination, "success": False, "error": "Cannot move to protected location"}
        
        # Check if destination exists
        if os.path.exists(destination):
            if not overwrite:
                return {"source": source, "destination": destination, "success": False, "error": "Destination already exists (use overwrite=true to replace)"}
            else:
                # Remove destination if overwrite is allowed
                if os.path.isfile(destination):
                    os.remove(destination)
                elif os.path.isdir(destination):
                    shutil.rmtree(destination)
        
        # Perform move
        shutil.move(source, destination)
        
        return {"source": source, "destination": destination, "success": True}
        
    except PermissionError:
        return {"source": source, "destination": destination, "success": False, "error": "Permission denied"}
    except Exception as e:
        return {"source": source, "destination": destination, "success": False, "error": str(e)}

@tool(
    name="delete_file",
    description="Delete a file or directory. Supports both absolute and relative paths. Requires recursive=true for directories. Cannot delete protected system files.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to delete. Can be absolute or relative (e.g., 'file.txt', './temp/data.json')"
            },
            "recursive": {
                "type": "boolean",
                "description": "Required to delete directories and their contents (default: false)"
            }
        },
        "required": ["path"]
    },
    key_param="path"
)
def delete_file(path: str, recursive: bool = False) -> dict:
    """
    Delete a file or directory.

    Args:
        path: Path to delete (absolute or relative)
        recursive: If True, delete directories and their contents (default: False)

    Safeguards:
        - Cannot delete protected system directories or files
        - Relative paths resolved to current working directory
        - Requires explicit recursive=True for directories

    Returns:
        Dict with success status
    """
    import shutil

    try:
        # Normalize path to absolute
        path = _normalize_path(path)
        
        # Check if exists
        if not os.path.exists(path):
            return {"path": path, "success": False, "error": "Path does not exist"}
        
        # Check if protected
        if _is_protected_path(path) or _is_protected_path(os.path.dirname(path)):
            return {"path": path, "success": False, "error": "Cannot delete protected system file or directory"}
        
        # Handle directory
        if os.path.isdir(path):
            if not recursive:
                return {"path": path, "success": False, "error": "Path is a directory (use recursive=true to delete)"}
            else:
                shutil.rmtree(path)
                return {"path": path, "success": True, "type": "directory", "recursive": True}
        
        # Handle file
        else:
            os.remove(path)
            return {"path": path, "success": True, "type": "file"}
        
    except PermissionError:
        return {"path": path, "success": False, "error": "Permission denied"}
    except Exception as e:
        return {"path": path, "success": False, "error": str(e)}

@tool(
    name="copy_file",
    description="Copy a file or directory to a new location. Supports both absolute and relative paths. Creates a duplicate.",
    parameters={
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "Source path. Can be absolute or relative (e.g., 'file.txt', './data/file.txt')"
            },
            "destination": {
                "type": "string",
                "description": "Destination path. Can be absolute or relative (e.g., 'backup.txt', './backup/file.txt')"
            },
            "overwrite": {
                "type": "boolean",
                "description": "Overwrite destination if it exists (default: false)"
            }
        },
        "required": ["source", "destination"]
    },
    key_param="source"
)
def copy_file(source: str, destination: str, overwrite: bool = False) -> dict:
    """
    Copy a file or directory.

    Args:
        source: Source path (absolute or relative)
        destination: Destination path (absolute or relative)
        overwrite: Whether to overwrite if destination exists (default: False)

    Safeguards:
        - Cannot copy from/to protected system directories
        - Relative paths resolved to current working directory

    Returns:
        Dict with success status
    """
    import shutil

    try:
        # Normalize paths to absolute
        source = _normalize_path(source)
        destination = _normalize_path(destination)
        
        # Check if source exists
        if not os.path.exists(source):
            return {"source": source, "destination": destination, "success": False, "error": "Source does not exist"}
        
        # Check protection
        if _is_protected_path(source):
            return {"source": source, "destination": destination, "success": False, "error": "Cannot copy from protected location"}
        
        dest_dir = os.path.dirname(destination)
        if _is_protected_path(destination) or _is_protected_path(dest_dir):
            return {"source": source, "destination": destination, "success": False, "error": "Cannot copy to protected location"}
        
        # Check if destination exists
        if os.path.exists(destination) and not overwrite:
            return {"source": source, "destination": destination, "success": False, "error": "Destination already exists (use overwrite=true to replace)"}
        
        # Perform copy
        if os.path.isfile(source):
            shutil.copy2(source, destination)
            copied_type = "file"
        elif os.path.isdir(source):
            if os.path.exists(destination):
                shutil.rmtree(destination)
            shutil.copytree(source, destination)
            copied_type = "directory"
        
        return {"source": source, "destination": destination, "success": True, "type": copied_type}
        
    except PermissionError:
        return {"source": source, "destination": destination, "success": False, "error": "Permission denied"}
    except Exception as e:
        return {"source": source, "destination": destination, "success": False, "error": str(e)}

@tool(
    name="get_current_directory",
    description="Get the current working directory of the server process.",
    parameters={
        "type": "object",
        "properties": {}
    },
    key_param=None
)
def get_current_directory() -> dict:
    """
    Get the current working directory.

    Returns:
        Dict with current directory path
    """
    try:
        cwd = os.getcwd()
        return {"current_directory": cwd, "success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@tool(
    name="calculate_directory_size",
    description="Calculate the total size of a directory and count files/subdirectories. Supports both absolute and relative paths. Useful for checking disk usage.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to analyze. Can be absolute or relative (e.g., 'C:\\projects', './data', or '.')"
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum depth to traverse (null for unlimited)"
            }
        },
        "required": ["path"]
    },
    key_param="path"
)
def calculate_directory_size(path: str, max_depth: int = None) -> dict:
    """
    Calculate total size of a directory and its contents.

    Args:
        path: Directory path (absolute or relative)
        max_depth: Maximum depth to traverse (None = unlimited)

    Safeguards:
        - Cannot calculate size of protected directories
        - Limited depth to prevent long operations

    Returns:
        Dict with total size and file count
    """
    try:
        # Normalize path to absolute
        path = _normalize_path(path)
        
        # Check if exists
        if not os.path.exists(path):
            return {"path": path, "exists": False, "error": "Path does not exist"}
        
        # Check if directory
        if not os.path.isdir(path):
            return {"path": path, "is_directory": False, "error": "Path is not a directory"}
        
        # Check if protected
        if _is_protected_path(path):
            return {"path": path, "protected": True, "error": "Cannot calculate size of protected directory"}
        
        total_size = 0
        file_count = 0
        dir_count = 0
        
        def _walk_directory(current_path, depth=0):
            nonlocal total_size, file_count, dir_count
            
            if max_depth is not None and depth > max_depth:
                return
            
            try:
                for entry in os.listdir(current_path):
                    entry_path = os.path.join(current_path, entry)
                    
                    # Skip protected subdirectories
                    if _is_protected_path(entry_path):
                        continue
                    
                    try:
                        if os.path.isfile(entry_path):
                            total_size += os.path.getsize(entry_path)
                            file_count += 1
                        elif os.path.isdir(entry_path):
                            dir_count += 1
                            _walk_directory(entry_path, depth + 1)
                    except (PermissionError, OSError):
                        continue
            except PermissionError:
                pass
        
        _walk_directory(path)
        
        return {
            "path": path,
            "total_size_bytes": total_size,
            "total_size_human": _format_size(total_size),
            "file_count": file_count,
            "directory_count": dir_count,
            "max_depth_used": max_depth
        }
        
    except Exception as e:
        return {"path": path, "error": str(e)}

@tool(
    name="find_in_files",
    description="Search for text within files (grep-like). Supports both absolute and relative paths. Useful for finding code, configuration values, or text across multiple files.",
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory to search in. Can be absolute or relative (e.g., 'C:\\projects', './src', or '.')"
            },
            "search_text": {
                "type": "string",
                "description": "Text to search for"
            },
            "file_pattern": {
                "type": "string",
                "description": "File pattern to search within (default: all files)"
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "Case-sensitive search (default: false)"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of matching files to return (default: 50)"
            }
        },
        "required": ["path", "search_text"]
    },
    key_param="search_text"
)
def find_in_files(path: str, search_text: str, file_pattern: str = "*", case_sensitive: bool = False, max_results: int = 50) -> dict:
    """
    Search for text within files (like grep).

    Args:
        path: Directory to search in (absolute or relative)
        search_text: Text to search for
        file_pattern: File pattern to search within (default: all files)
        case_sensitive: Whether search is case-sensitive (default: False)
        max_results: Maximum number of matching files to return (default: 50)

    Safeguards:
        - Cannot search protected directories
        - Limited to text files only
        - Limited results to prevent overwhelming output

    Returns:
        Dict with matching files and line numbers
    """
    import fnmatch

    try:
        # Normalize path to absolute
        path = _normalize_path(path)
        
        # Check if exists and is directory
        if not os.path.exists(path):
            return {"path": path, "exists": False, "error": "Path does not exist"}
        
        if not os.path.isdir(path):
            return {"path": path, "is_directory": False, "error": "Path is not a directory"}
        
        # Check if protected
        if _is_protected_path(path):
            return {"path": path, "protected": True, "error": "Cannot search protected directory"}
        
        # Prepare search
        if not case_sensitive:
            search_text = search_text.lower()
        
        matches = []
        
        for root, dirs, files in os.walk(path):
            # Skip protected subdirectories
            dirs[:] = [d for d in dirs if not _is_protected_path(os.path.join(root, d))]
            
            for filename in files:
                if not fnmatch.fnmatch(filename, file_pattern):
                    continue
                
                file_path = os.path.join(root, filename)
                
                try:
                    # Try to read as text file
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        matching_lines = []
                        for line_num, line in enumerate(f, 1):
                            check_line = line if case_sensitive else line.lower()
                            if search_text in check_line:
                                matching_lines.append({
                                    "line_number": line_num,
                                    "content": line.rstrip()
                                })
                        
                        if matching_lines:
                            matches.append({
                                "file": file_path,
                                "matches": matching_lines,
                                "match_count": len(matching_lines)
                            })
                            
                            if len(matches) >= max_results:
                                break
                
                except (PermissionError, OSError, UnicodeDecodeError):
                    # Skip files we can't read or aren't text
                    continue
            
            if len(matches) >= max_results:
                break
        
        return {
            "path": path,
            "search_text": search_text,
            "file_pattern": file_pattern,
            "case_sensitive": case_sensitive,
            "results": matches,
            "total_files_found": len(matches),
            "truncated": len(matches) >= max_results
        }

    except Exception as e:
        return {"path": path, "search_text": search_text, "error": str(e)}

# =============================================================================
# WEB TOOLS - SECURITY UTILITIES
# =============================================================================

def sanitize_web_content(text: str, aggressive: bool = True) -> str:
    """
    Sanitize web content to prevent prompt injection attacks.

    This function removes or neutralizes patterns that could be used to
    manipulate the LLM through malicious web content.

    Args:
        text: Raw text extracted from webpage
        aggressive: If True, apply stricter filtering (default: True)

    Returns:
        Sanitized text safe for LLM consumption

    Security measures:
    - Removes role-like prefixes (USER:, ASSISTANT:, SYSTEM:, etc.)
    - Filters instruction-like patterns
    - Removes excessive whitespace/newlines
    - Neutralizes common prompt injection attempts
    - Removes suspicious Unicode characters
    """
    import re

    if not text:
        return text

    # 1. Remove common role prefixes that could confuse the LLM
    # Matches patterns like "USER:", "ASSISTANT:", "SYSTEM:", "Human:", "AI:", etc.
    role_patterns = [
        r'^\s*(USER|HUMAN|PERSON):\s*',
        r'^\s*(ASSISTANT|AI|BOT|CLAUDE|GPT):\s*',
        r'^\s*(SYSTEM|INSTRUCTION|ADMIN|ROOT):\s*',
        r'\n\s*(USER|HUMAN|PERSON):\s*',
        r'\n\s*(ASSISTANT|AI|BOT|CLAUDE|GPT):\s*',
        r'\n\s*(SYSTEM|INSTRUCTION|ADMIN|ROOT):\s*',
    ]

    for pattern in role_patterns:
        text = re.sub(pattern, '\n', text, flags=re.IGNORECASE | re.MULTILINE)

    # 2. Remove instruction-like patterns
    if aggressive:
        # Patterns that look like system instructions
        instruction_patterns = [
            r'\[INST(?:RUCTION)?\].*?\[/INST(?:RUCTION)?\]',  # [INSTRUCTION] tags
            r'<\|.*?\|>',  # Special tokens like <|im_start|>
            r'<<SYS>>.*?<</SYS>>',  # Llama-style system tags
            r'\[SYSTEM\].*?\[/SYSTEM\]',  # System message tags
            r'```(?:system|instruction|prompt).*?```',  # Code blocks with suspicious labels
        ]

        for pattern in instruction_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

    # 3. Remove attempts to break out of context
    breakout_patterns = [
        r'Ignore (?:all )?previous (?:instructions|commands|prompts)',
        r'Disregard (?:all )?(?:above|previous|prior)',
        r'Forget (?:all )?(?:previous|prior) (?:instructions|commands)',
        r'New (?:instruction|command|directive|task):',
        r'Override (?:previous )?(?:instructions|settings)',
        r'IMPORTANT:.*?(?:you must|you should|execute|run)',
    ]

    for pattern in breakout_patterns:
        text = re.sub(pattern, '[filtered content]', text, flags=re.IGNORECASE)

    # 4. Normalize whitespace (prevents hidden instructions via Unicode spaces)
    # Replace various Unicode whitespace with regular space
    text = re.sub(r'[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000\uFEFF]+', ' ', text)

    # 5. Remove excessive newlines (could be used to create fake chat transcripts)
    text = re.sub(r'\n{4,}', '\n\n\n', text)  # Max 3 consecutive newlines

    # 6. Remove zero-width characters (could hide instructions)
    zero_width_chars = [
        '\u200B',  # Zero-width space
        '\u200C',  # Zero-width non-joiner
        '\u200D',  # Zero-width joiner
        '\u2060',  # Word joiner
        '\uFEFF',  # Zero-width no-break space
    ]
    for char in zero_width_chars:
        text = text.replace(char, '')

    # 7. Limit excessive repetition (could be used for token exhaustion attacks)
    # Replace 10+ repeated characters with just 3
    text = re.sub(r'(.)\1{9,}', r'\1\1\1', text)

    # 8. Remove content that looks like prompt delimiters
    delimiter_patterns = [
        r'={10,}',  # Long sequences of equals signs
        r'-{10,}',  # Long sequences of dashes
        r'#{5,}',   # Multiple hash symbols
    ]

    for pattern in delimiter_patterns:
        text = re.sub(pattern, '---', text)

    return text.strip()


# =============================================================================
# WEB TOOLS
# =============================================================================

@tool(
    name="web_search",
    description="Search the web for current information using DuckDuckGo. Use this when you need up-to-date information, facts, news, or answers that are beyond your knowledge cutoff. Returns titles, snippets, and URLs.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query (e.g., 'Python 3.12 new features', 'current weather NYC', 'latest AI news')"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of search results to return (default: 5, max: 10)"
            }
        },
        "required": ["query"]
    },
    key_param="query"
)
def web_search(query: str, max_results: int = 5) -> dict:
    """
    Search the web using DuckDuckGo.

    Args:
        query: Search query string
        max_results: Maximum number of results (default: 5, max: 10)

    Returns:
        Dict with search results containing title, snippet, and URL
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        # Try new package name first (ddgs), fall back to old (duckduckgo-search)
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        import server_config as config

        logger.info(f"Web search initiated: query='{query}', max_results={max_results}")

        # Limit max_results to prevent overwhelming context
        max_results = min(max_results, 10)

        # Perform search with timeout
        results = []
        try:
            # Create DDGS instance with timeout
            ddgs = DDGS(timeout=20)

            # Get search results (returns a generator/list depending on version)
            search_results = ddgs.text(query, max_results=max_results)

            logger.debug(f"Search results type: {type(search_results)}")

            # Handle both generator and list returns
            if search_results is None:
                logger.warning("DDGS returned None")
                return {
                    "query": query,
                    "results": [],
                    "count": 0,
                    "success": False,
                    "error": "DuckDuckGo returned no results. Try a different query or wait a moment."
                }

            # Iterate through results
            for idx, result in enumerate(search_results):
                if idx >= max_results:
                    break

                logger.debug(f"Result {idx}: {result}")

                title = result.get("title", "")
                snippet = result.get("body", "")
                url = result.get("href", "")

                if not url:  # Skip results without URLs
                    logger.warning(f"Skipping result {idx} - no URL")
                    continue

                # Sanitize title and snippet to prevent prompt injection
                if config.WEB_CONTENT_SANITIZATION:
                    aggressive = config.WEB_CONTENT_AGGRESSIVE_SANITIZATION
                    title = sanitize_web_content(title, aggressive=aggressive)
                    snippet = sanitize_web_content(snippet, aggressive=aggressive)

                results.append({
                    "title": title,
                    "snippet": snippet,
                    "url": url,
                })

            logger.info(f"Found {len(results)} results for query: '{query}'")

        except Exception as search_error:
            logger.error(f"DDGS search error: {type(search_error).__name__}: {search_error}", exc_info=True)
            return {
                "query": query,
                "results": [],
                "count": 0,
                "success": False,
                "error": f"Search API error: {type(search_error).__name__}: {str(search_error)}"
            }

        # Check if we got any results
        if not results:
            logger.warning(f"No results found for query: '{query}'")
            return {
                "query": query,
                "results": [],
                "count": 0,
                "success": False,
                "error": "No results found. Try rephrasing your query or check internet connectivity."
            }

        # Wrap results in clear delimiters for LLM safety
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append({
                "title": result["title"],
                "snippet": f"<web_search_result>\n{result['snippet']}\n</web_search_result>",
                "url": result["url"]
            })

        return {
            "query": query,
            "results": formatted_results,
            "count": len(formatted_results),
            "success": True,
            "security_note": "Content has been sanitized to prevent prompt injection attacks" if config.WEB_CONTENT_SANITIZATION else None
        }

    except ImportError as e:
        logger.error(f"Import error: {e}")
        return {
            "query": query,
            "results": [],
            "count": 0,
            "success": False,
            "error": "ddgs library not installed. Install with: pip install ddgs (or pip install duckduckgo-search for older version)"
        }
    except Exception as e:
        logger.error(f"Unexpected error in web_search: {type(e).__name__}: {e}", exc_info=True)
        return {
            "query": query,
            "results": [],
            "count": 0,
            "success": False,
            "error": f"Search failed: {type(e).__name__}: {str(e)}"
        }

@tool(
    name="read_webpage",
    description="Fetch and extract the main text content from a webpage URL. Use this to read articles, documentation, or any web content. Returns clean text without HTML markup.",
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The full URL to fetch (must start with http:// or https://)"
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters to return (default: 3000, helps with context limits)"
            }
        },
        "required": ["url"]
    },
    key_param="url"
)
def read_webpage(url: str, max_chars: int = 3000) -> dict:
    """
    Fetch and extract text content from a webpage.

    Args:
        url: URL to fetch
        max_chars: Maximum characters to return (default: 3000)

    Returns:
        Dict with webpage content and metadata
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        import server_config as config

        # Validate URL
        if not url.startswith(('http://', 'https://')):
            return {
                "url": url,
                "content": "",
                "success": False,
                "error": "URL must start with http:// or https://"
            }

        # Fetch webpage
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Parse HTML
        soup = BeautifulSoup(response.content, 'lxml')

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Get text
        text = soup.get_text(separator='\n', strip=True)

        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = '\n'.join(lines)

        # Sanitize content BEFORE truncation to prevent injection in truncated content
        if config.WEB_CONTENT_SANITIZATION:
            aggressive = config.WEB_CONTENT_AGGRESSIVE_SANITIZATION
            clean_text = sanitize_web_content(clean_text, aggressive=aggressive)

        # Truncate if needed (after sanitization to get accurate length)
        was_truncated = False
        if len(clean_text) > max_chars:
            clean_text = clean_text[:max_chars]
            was_truncated = True

        # Get title and sanitize it too
        title = soup.title.string if soup.title else "No title"
        if config.WEB_CONTENT_SANITIZATION:
            title = sanitize_web_content(title, aggressive=aggressive)

        # Wrap content in clear delimiters to help LLM understand it's external content
        wrapped_content = f"""<webpage_content source="{url}">
{clean_text}
</webpage_content>"""

        if was_truncated:
            wrapped_content += f"\n\n[Content truncated at {max_chars} characters for token limit]"

        return {
            "url": url,
            "title": title,
            "content": wrapped_content,
            "length": len(clean_text),
            "truncated": was_truncated,
            "success": True,
            "security_note": "Content has been sanitized to prevent prompt injection attacks" if config.WEB_CONTENT_SANITIZATION else None
        }

    except requests.exceptions.Timeout:
        return {
            "url": url,
            "content": "",
            "success": False,
            "error": "Request timed out after 10 seconds"
        }
    except requests.exceptions.RequestException as e:
        return {
            "url": url,
            "content": "",
            "success": False,
            "error": f"Failed to fetch URL: {str(e)}"
        }
    except ImportError as e:
        missing_lib = "requests" if "requests" in str(e) else "beautifulsoup4"
        return {
            "url": url,
            "content": "",
            "success": False,
            "error": f"{missing_lib} library not installed. Install with: pip install {missing_lib}"
        }
    except Exception as e:
        return {
            "url": url,
            "content": "",
            "success": False,
            "error": f"Error reading webpage: {str(e)}"
        }
