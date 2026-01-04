"""
tools.py - Native tools for AI Lab Server
"""

import socket
import subprocess
import platform
import time
import os
from typing import Dict, Optional, List
from pathlib import Path

# =============================================================================
# NETWORKING TOOLS
# =============================================================================

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

def read_file(path: str) -> dict:
    """
    Read and return the contents of a file.
    
    Args:
        path: The file path to read from
        
    Returns:
        Dict with path, lines (content), and optional error
    """
    try:
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

def write_file(path: str, lines: str) -> dict:
    """
    Write content to a file at the specified path.
    
    Args:
        path: The absolute path of the file to write to
        lines: The content to write (string or list of strings)
        
    Safeguards:
        1. Path must be absolute
        2. Parent directory must exist
        3. Cannot write to core system directories
        
    Returns:
        Dict with status and optional error
    """
    try:
        # Ensure the path is absolute
        if not os.path.isabs(path):
            return {
                "path": path,
                "success": False,
                "error": "Path must be absolute"
            }
        
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

def list_contents(path: str, show_hidden: bool = False) -> dict:
    """
    List files and directories in the specified path.
    
    Args:
        path: Directory path to list contents of
        show_hidden: Whether to show hidden files (default: False)
        
    Safeguards:
        - Cannot list root directories (C:\ on Windows, / on Linux)
        - Cannot list protected system directories
        
    Returns:
        Dict with files, directories, and optional error
    """
    try:
        # Normalize path
        path = os.path.abspath(path)
        
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

def search_files(path: str, pattern: str, recursive: bool = True, max_results: int = 100) -> dict:
    """
    Search for files matching a pattern in the specified directory.
    
    Args:
        path: Directory to search in
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
        # Normalize path
        path = os.path.abspath(path)
        
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

def get_file_info(path: str) -> dict:
    """
    Get metadata about a file or directory without reading its contents.
    
    Args:
        path: Path to the file or directory
        
    Returns:
        Dict with size, modification time, type, permissions
    """
    try:
        # Normalize path
        path = os.path.abspath(path)
        
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

def create_directory(path: str, parents: bool = True) -> dict:
    """
    Create a new directory.
    
    Args:
        path: Directory path to create
        parents: Create parent directories if needed (default: True)
        
    Safeguards:
        - Cannot create directories in protected system locations
        - Path must be absolute
        
    Returns:
        Dict with success status
    """
    try:
        # Ensure absolute path
        if not os.path.isabs(path):
            return {"path": path, "success": False, "error": "Path must be absolute"}
        
        # Normalize path
        path = os.path.abspath(path)
        
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

def move_file(source: str, destination: str, overwrite: bool = False) -> dict:
    """
    Move or rename a file or directory.
    
    Args:
        source: Source path
        destination: Destination path
        overwrite: Whether to overwrite if destination exists (default: False)
        
    Safeguards:
        - Cannot move from/to protected system directories
        - Paths must be absolute
        - Won't overwrite unless explicitly allowed
        
    Returns:
        Dict with success status
    """
    import shutil
    
    try:
        # Ensure absolute paths
        if not os.path.isabs(source) or not os.path.isabs(destination):
            return {"source": source, "destination": destination, "success": False, "error": "Paths must be absolute"}
        
        # Normalize paths
        source = os.path.abspath(source)
        destination = os.path.abspath(destination)
        
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

def delete_file(path: str, recursive: bool = False) -> dict:
    """
    Delete a file or directory.
    
    Args:
        path: Path to delete
        recursive: If True, delete directories and their contents (default: False)
        
    Safeguards:
        - Cannot delete protected system directories or files
        - Path must be absolute
        - Requires explicit recursive=True for directories
        
    Returns:
        Dict with success status
    """
    import shutil
    
    try:
        # Ensure absolute path
        if not os.path.isabs(path):
            return {"path": path, "success": False, "error": "Path must be absolute"}
        
        # Normalize path
        path = os.path.abspath(path)
        
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

def copy_file(source: str, destination: str, overwrite: bool = False) -> dict:
    """
    Copy a file or directory.
    
    Args:
        source: Source path
        destination: Destination path
        overwrite: Whether to overwrite if destination exists (default: False)
        
    Safeguards:
        - Cannot copy from/to protected system directories
        - Paths must be absolute
        
    Returns:
        Dict with success status
    """
    import shutil
    
    try:
        # Ensure absolute paths
        if not os.path.isabs(source) or not os.path.isabs(destination):
            return {"source": source, "destination": destination, "success": False, "error": "Paths must be absolute"}
        
        # Normalize paths
        source = os.path.abspath(source)
        destination = os.path.abspath(destination)
        
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

def calculate_directory_size(path: str, max_depth: int = None) -> dict:
    """
    Calculate total size of a directory and its contents.
    
    Args:
        path: Directory path
        max_depth: Maximum depth to traverse (None = unlimited)
        
    Safeguards:
        - Cannot calculate size of protected directories
        - Limited depth to prevent long operations
        
    Returns:
        Dict with total size and file count
    """
    try:
        # Normalize path
        path = os.path.abspath(path)
        
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

def find_in_files(path: str, search_text: str, file_pattern: str = "*", case_sensitive: bool = False, max_results: int = 50) -> dict:
    """
    Search for text within files (like grep).
    
    Args:
        path: Directory to search in
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
        # Normalize path
        path = os.path.abspath(path)
        
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
# TOOL REGISTRY
# =============================================================================

def get_available_tools() -> list:
    """Return list of available tool definitions for the LLM"""
    return [
        {
            "name": "lookup_hostname",
            "description": "Look up the IP address of a hostname/domain and check if it's reachable. Use this when asked about current IPs, domain resolution, or network connectivity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hostname": {
                        "type": "string",
                        "description": "The hostname or domain name to look up (e.g., 'google.com', 'github.com')"
                    }
                },
                "required": ["hostname"]
            }
        },
        {
            "name": "measure_http_latency",
            "description": "Determine the latency time required to access a domain name via the HTTPS protocol",
            "parameters": {
                "type": "object",
                "properties": {
                    "hostname": {
                        "type": "string",
                        "description": "The hostname or domain name to test latency for (e.g., 'google.com', 'github.com')"
                    }
                },
                "required": ["hostname"]
            }
        },
        {
            "name": "read_file",
            "description": "Read and return the contents of a text file. Use this to examine code, configuration files, logs, or any text-based files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The absolute file path to read from (e.g., 'C:\\Users\\user\\file.txt' or '/home/user/file.txt')"
                    }
                },
                "required": ["path"]
            }
        },
        {
            "name": "write_file",
            "description": "Create or overwrite a file with the supplied content. Safeguards prevent writing to system directories. Use this to create new files or update existing ones.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The absolute file path to write to (e.g., 'C:\\Users\\user\\output.txt' or '/home/user/output.txt')"
                    },
                    "lines": {
                        "type": "string",
                        "description": "The content to write to the file (as a string)"
                    }
                },
                "required": ["path", "lines"]
            }
        },
        {
            "name": "list_contents",
            "description": "List all files and directories in a specified directory. Returns files with sizes and directories. Protected system directories cannot be listed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The absolute directory path to list contents of (e.g., 'C:\\Users\\user\\Documents' or '/home/user/projects')"
                    },
                    "show_hidden": {
                        "type": "boolean",
                        "description": "Whether to include hidden files and directories (default: false)"
                    }
                },
                "required": ["path"]
            }
        },
        {
            "name": "search_files",
            "description": "Search for files matching a pattern within a directory. Supports wildcards like *.py, test*.txt. Useful for finding specific files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The directory path to search in"
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
            }
        },
        {
            "name": "get_file_info",
            "description": "Get metadata about a file or directory without reading its contents. Returns size, modification date, type, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file or directory"
                    }
                },
                "required": ["path"]
            }
        },
        {
            "name": "create_directory",
            "description": "Create a new directory. Can create parent directories if needed. Cannot create in protected system locations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path for the new directory"
                    },
                    "parents": {
                        "type": "boolean",
                        "description": "Create parent directories if they don't exist (default: true)"
                    }
                },
                "required": ["path"]
            }
        },
        {
            "name": "move_file",
            "description": "Move or rename a file or directory. Can be used to reorganize files or rename them.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Source path (absolute)"
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination path (absolute)"
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Overwrite destination if it exists (default: false)"
                    }
                },
                "required": ["source", "destination"]
            }
        },
        {
            "name": "delete_file",
            "description": "Delete a file or directory. Requires recursive=true for directories. Cannot delete protected system files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to delete"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Required to delete directories and their contents (default: false)"
                    }
                },
                "required": ["path"]
            }
        },
        {
            "name": "copy_file",
            "description": "Copy a file or directory to a new location. Creates a duplicate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Source path (absolute)"
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination path (absolute)"
                    },
                    "overwrite": {
                        "type": "boolean",
                        "description": "Overwrite destination if it exists (default: false)"
                    }
                },
                "required": ["source", "destination"]
            }
        },
        {
            "name": "get_current_directory",
            "description": "Get the current working directory of the server process.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        },
        {
            "name": "calculate_directory_size",
            "description": "Calculate the total size of a directory and count files/subdirectories. Useful for checking disk usage.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to analyze"
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum depth to traverse (null for unlimited)"
                    }
                },
                "required": ["path"]
            }
        },
        {
            "name": "find_in_files",
            "description": "Search for text within files (grep-like). Useful for finding code, configuration values, or text across multiple files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to search in"
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
            }
        }
    ]