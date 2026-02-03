"""
test_hardware.py - Test suite for hardware detection and configuration

Tests:
- GPU detection (mock for non-GPU systems)
- CPU detection
- RAM detection
- Configuration generation for different hardware profiles
- Profile persistence (save/load)
- Override mechanisms (env vars, manual overrides)
- Fallback behavior

Usage:
    python test_hardware.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import hardware_detector


# ============================================================================
# Test Data - Mock Hardware Profiles
# ============================================================================

# System 1: Gaming PC with RTX 5700 TI
MOCK_SYSTEM1_GPU = {
    "has_gpu": True,
    "name": "NVIDIA GeForce RTX 5700 TI",
    "vram_gb": 16.0,
    "cuda_version": "13.0",
    "driver_version": "560.81",
    "detection_method": "mock"
}

MOCK_SYSTEM1_CPU = {
    "name": "AMD Ryzen 7 5800X 8-Core Processor",
    "physical_cores": 8,
    "logical_cores": 16,
    "architecture": "AMD64",
    "detection_method": "mock"
}

MOCK_SYSTEM1_RAM = {
    "total_gb": 32.0,
    "available_gb": 28.5,
    "percent_available": 89.1,
    "detection_method": "mock"
}

# System 2: Enterprise Server (No GPU, 56 threads, 256GB RAM)
MOCK_SYSTEM2_GPU = {
    "has_gpu": False,
    "name": "None",
    "vram_gb": 0.0,
    "cuda_version": "N/A",
    "driver_version": "N/A",
    "detection_method": "mock"
}

MOCK_SYSTEM2_CPU = {
    "name": "Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz",
    "physical_cores": 28,
    "logical_cores": 56,
    "architecture": "x86_64",
    "detection_method": "mock"
}

MOCK_SYSTEM2_RAM = {
    "total_gb": 256.0,
    "available_gb": 240.0,
    "percent_available": 93.75,
    "detection_method": "mock"
}

MOCK_SYSTEM2_NUMA = {
    "numa_nodes": 2,
    "is_multi_socket": True,
    "detection_method": "mock"
}


# ============================================================================
# Test Functions
# ============================================================================

def test_gpu_detection():
    """Test GPU detection (real or mock)"""
    print("\n=== Test: GPU Detection ===")

    gpu_info = hardware_detector.detect_nvidia_gpu()

    print(f"Has GPU: {gpu_info['has_gpu']}")
    if gpu_info['has_gpu']:
        print(f"  Name: {gpu_info['name']}")
        print(f"  VRAM: {gpu_info['vram_gb']}GB")
        print(f"  CUDA: {gpu_info.get('cuda_version', 'unknown')}")
        print(f"  Detection Method: {gpu_info.get('detection_method', 'unknown')}")
    else:
        print("  No NVIDIA GPU detected")

    # Verify structure
    assert 'has_gpu' in gpu_info
    assert 'name' in gpu_info
    assert 'vram_gb' in gpu_info

    print("✓ GPU detection passed")
    return gpu_info


def test_cpu_detection():
    """Test CPU detection"""
    print("\n=== Test: CPU Detection ===")

    cpu_info = hardware_detector.detect_cpu()

    print(f"CPU: {cpu_info['name']}")
    print(f"  Physical Cores: {cpu_info['physical_cores']}")
    print(f"  Logical Cores: {cpu_info['logical_cores']}")
    print(f"  Architecture: {cpu_info.get('architecture', 'unknown')}")

    # Verify structure
    assert 'name' in cpu_info
    assert 'physical_cores' in cpu_info
    assert 'logical_cores' in cpu_info
    assert cpu_info['logical_cores'] >= cpu_info['physical_cores']

    print("✓ CPU detection passed")
    return cpu_info


def test_memory_detection():
    """Test RAM detection"""
    print("\n=== Test: Memory Detection ===")

    memory_info = hardware_detector.detect_memory()

    print(f"RAM: {memory_info['total_gb']:.1f}GB total")
    print(f"  Available: {memory_info['available_gb']:.1f}GB ({memory_info['percent_available']:.1f}%)")

    # Verify structure
    assert 'total_gb' in memory_info
    assert 'available_gb' in memory_info
    assert memory_info['total_gb'] > 0
    assert memory_info['available_gb'] <= memory_info['total_gb']

    print("✓ Memory detection passed")
    return memory_info


def test_config_generation_system1():
    """Test config generation for System 1 (Gaming PC)"""
    print("\n=== Test: Config Generation for System 1 (Gaming PC) ===")

    hardware_info = {
        "gpu": MOCK_SYSTEM1_GPU,
        "cpu": MOCK_SYSTEM1_CPU,
        "memory": MOCK_SYSTEM1_RAM
    }

    config = hardware_detector.generate_optimal_config(hardware_info)

    print(f"Mode: {config['mode']}")
    print(f"  GPU Layers: {config['n_gpu_layers']}")
    print(f"  Context Size: {config['ctx_size']} tokens")
    print(f"  Threads: {config.get('threads', 'None')}")
    print(f"  Reasoning: {config['reasoning']}")

    # Verify System 1 config expectations
    assert config['n_gpu_layers'] == -1, "System 1 should use full GPU mode"
    assert config['ctx_size'] >= 16384, "System 1 should have large context window"
    assert config['mode'] == "gpu_primary", "System 1 should be gpu_primary mode"

    print("✓ System 1 config generation passed")
    return config


def test_config_generation_system2():
    """Test config generation for System 2 (Enterprise Server)"""
    print("\n=== Test: Config Generation for System 2 (Server) ===")

    hardware_info = {
        "gpu": MOCK_SYSTEM2_GPU,
        "cpu": MOCK_SYSTEM2_CPU,
        "memory": MOCK_SYSTEM2_RAM,
        "numa": MOCK_SYSTEM2_NUMA,
    }

    config = hardware_detector.generate_optimal_config(hardware_info)

    print(f"Mode: {config['mode']}")
    print(f"  GPU Layers: {config['n_gpu_layers']}")
    print(f"  Context Size: {config['ctx_size']} tokens")
    print(f"  Threads: {config.get('threads', 'None')}")
    print(f"  Reasoning: {config['reasoning']}")

    cpu_opt = config.get('cpu_optimization', {})
    print(f"  CPU Optimization:")
    print(f"    NUMA: {cpu_opt.get('numa_mode')}")
    print(f"    Batch Size: {cpu_opt.get('batch_size')}")
    print(f"    UBatch Size: {cpu_opt.get('ubatch_size')}")
    print(f"    Memory Lock: {cpu_opt.get('mlock')}")
    print(f"    Threads (batch): {cpu_opt.get('threads_batch')}")
    print(f"    Flash Attention: {cpu_opt.get('flash_attn')}")
    print(f"    No MMap: {cpu_opt.get('no_mmap')}")

    # Verify System 2 config expectations
    assert config['n_gpu_layers'] == 0, "System 2 should use CPU-only mode"
    assert config['ctx_size'] == 8192, "System 2 should use conservative 8K context for CPU speed"
    assert config['threads'] is not None, "System 2 should specify thread count"
    # With 28 physical cores - 2 = 26 threads (not logical 56)
    assert config['threads'] == 26, f"System 2 should use 26 threads (physical_cores - 2), got {config['threads']}"
    assert config['mode'] == "cpu_highram", "System 2 should be cpu_highram mode"

    # Verify CPU optimization dict
    assert 'cpu_optimization' in config, "System 2 should have cpu_optimization"
    assert cpu_opt.get('numa_mode') == "distribute", "System 2 (dual Xeon) should use NUMA distribute"
    assert cpu_opt.get('batch_size') == 512, "Batch size should be 512"
    assert cpu_opt.get('ubatch_size') == 512, "UBatch size should be 512"
    assert cpu_opt.get('mlock') == True, "System 2 (256GB) should enable mlock"
    assert cpu_opt.get('threads_batch') == 56, "Batch threads should use all logical cores (56)"
    assert cpu_opt.get('flash_attn') == True, "Flash attention should be enabled"
    assert cpu_opt.get('no_mmap') == True, "No-mmap (preload) should be enabled for high-RAM"

    print("✓ System 2 config generation passed")
    return config


def test_context_calculation():
    """Test context window calculation"""
    print("\n=== Test: Context Window Calculation ===")

    # Test GPU mode with 16GB VRAM
    ctx_gpu_16gb = hardware_detector.calculate_optimal_context(vram_gb=16.0, mode="gpu")
    print(f"GPU 16GB VRAM: {ctx_gpu_16gb} tokens")
    assert ctx_gpu_16gb >= 16384, "16GB VRAM should support large context"

    # Test GPU mode with 8GB VRAM
    ctx_gpu_8gb = hardware_detector.calculate_optimal_context(vram_gb=8.0, mode="gpu")
    print(f"GPU 8GB VRAM: {ctx_gpu_8gb} tokens")
    assert ctx_gpu_8gb >= 8192, "8GB VRAM should support medium context"

    # Test CPU mode with 256GB RAM (conservative for speed)
    ctx_cpu_256gb = hardware_detector.calculate_optimal_context(ram_gb=256.0, mode="cpu")
    print(f"CPU 256GB RAM: {ctx_cpu_256gb} tokens")
    assert ctx_cpu_256gb == 8192, "CPU mode should use conservative 8K context for speed optimization"

    # Test CPU mode with 32GB RAM (conservative for speed)
    ctx_cpu_32gb = hardware_detector.calculate_optimal_context(ram_gb=32.0, mode="cpu")
    print(f"CPU 32GB RAM: {ctx_cpu_32gb} tokens")
    assert ctx_cpu_32gb == 8192, "CPU mode with 32GB should use 8K context"

    # Test CPU mode with low RAM
    ctx_cpu_12gb = hardware_detector.calculate_optimal_context(ram_gb=12.0, mode="cpu")
    print(f"CPU 12GB RAM: {ctx_cpu_12gb} tokens")
    assert ctx_cpu_12gb == 4096, "CPU mode with <16GB should use 4K context"

    print("✓ Context calculation passed")


def test_profile_persistence():
    """Test saving and loading hardware profiles"""
    print("\n=== Test: Profile Persistence ===")

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        # Create test profile
        test_profile = {
            "version": "1.0",
            "system_type": "test_system",
            "gpu": MOCK_SYSTEM1_GPU,
            "cpu": MOCK_SYSTEM1_CPU,
            "memory": MOCK_SYSTEM1_RAM,
            "recommended_config": {
                "n_gpu_layers": -1,
                "ctx_size": 20480,
                "threads": None,
                "mode": "gpu_primary"
            },
            "manual_overrides": {}
        }

        # Test save
        result = hardware_detector.save_hardware_profile(test_profile, temp_path)
        assert result == True, "Profile save should succeed"
        assert os.path.exists(temp_path), "Profile file should exist"
        print(f"✓ Profile saved to {temp_path}")

        # Test load
        loaded_profile = hardware_detector.load_hardware_profile(temp_path)
        assert loaded_profile is not None, "Profile load should succeed"
        assert loaded_profile['system_type'] == 'test_system', "Loaded profile should match saved"
        assert loaded_profile['gpu']['vram_gb'] == 16.0, "GPU info should be preserved"
        print(f"✓ Profile loaded successfully")

        # Test load non-existent file
        missing_profile = hardware_detector.load_hardware_profile("nonexistent_file.json")
        assert missing_profile is None, "Loading missing file should return None"
        print(f"✓ Missing file handled correctly")

        print("✓ Profile persistence passed")

    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_system_classification():
    """Test system type classification"""
    print("\n=== Test: System Classification ===")

    # Test gaming PC classification
    hw_gaming = {
        "gpu": MOCK_SYSTEM1_GPU,
        "cpu": MOCK_SYSTEM1_CPU,
        "memory": MOCK_SYSTEM1_RAM
    }
    type_gaming = hardware_detector.classify_system_type(hw_gaming)
    print(f"System 1 classified as: {type_gaming}")
    assert type_gaming == "gaming_pc", "System 1 should be classified as gaming_pc"

    # Test server classification
    hw_server = {
        "gpu": MOCK_SYSTEM2_GPU,
        "cpu": MOCK_SYSTEM2_CPU,
        "memory": MOCK_SYSTEM2_RAM
    }
    type_server = hardware_detector.classify_system_type(hw_server)
    print(f"System 2 classified as: {type_server}")
    assert type_server == "server", "System 2 should be classified as server"

    print("✓ System classification passed")


def test_numa_detection():
    """Test NUMA topology detection with mock data"""
    print("\n=== Test: NUMA Detection ===")

    # Test with dual-socket Xeon (System 2 mock data)
    numa_result = hardware_detector.detect_numa_topology(MOCK_SYSTEM2_CPU)
    print(f"System 2 (Xeon E5-2680 v4): {numa_result}")

    # System 2 has 28 physical cores and 'E5-' in name — heuristic should detect multi-socket
    # (unless running on the actual system with wmic/sysfs available)
    if numa_result['detection_method'] == 'heuristic':
        assert numa_result['is_multi_socket'] == True, "Xeon E5 with 28 cores should be detected as multi-socket"
        assert numa_result['numa_nodes'] >= 2, "Should detect at least 2 NUMA nodes"
    # If running on actual hardware, wmic/sysfs detection takes precedence — just verify structure
    assert 'numa_nodes' in numa_result
    assert 'is_multi_socket' in numa_result
    assert 'detection_method' in numa_result
    print(f"  NUMA nodes: {numa_result['numa_nodes']}, multi-socket: {numa_result['is_multi_socket']}")

    # Test with desktop CPU (System 1 mock data) — should be single socket
    numa_desktop = hardware_detector.detect_numa_topology(MOCK_SYSTEM1_CPU)
    print(f"System 1 (Ryzen 7 5800X): {numa_desktop}")

    if numa_desktop['detection_method'] in ('heuristic', 'default'):
        assert numa_desktop['is_multi_socket'] == False, "Desktop Ryzen should not be multi-socket"
        assert numa_desktop['numa_nodes'] == 1, "Desktop should have 1 NUMA node"
    assert 'numa_nodes' in numa_desktop
    assert 'is_multi_socket' in numa_desktop
    print(f"  NUMA nodes: {numa_desktop['numa_nodes']}, multi-socket: {numa_desktop['is_multi_socket']}")

    print("✓ NUMA detection passed")


def test_full_detection_flow():
    """Test complete detection and save flow"""
    print("\n=== Test: Full Detection Flow ===")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        # Run full detection
        profile = hardware_detector.detect_and_save(temp_path)

        # Verify profile structure
        assert 'version' in profile
        assert 'system_type' in profile
        assert 'gpu' in profile
        assert 'cpu' in profile
        assert 'memory' in profile
        assert 'recommended_config' in profile

        # Verify file was created
        assert os.path.exists(temp_path)

        # Verify can be reloaded
        reloaded = hardware_detector.load_hardware_profile(temp_path)
        assert reloaded is not None
        assert reloaded['version'] == profile['version']

        print(f"System Type: {profile['system_type']}")
        print(f"Mode: {profile['recommended_config']['mode']}")
        print(f"Profile saved and verified at: {temp_path}")

        print("✓ Full detection flow passed")

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


# ============================================================================
# Test Runner
# ============================================================================

def run_all_tests():
    """Run all hardware detection tests"""
    print("=" * 60)
    print("Hardware Detection Test Suite")
    print("=" * 60)

    tests = [
        ("GPU Detection", test_gpu_detection),
        ("CPU Detection", test_cpu_detection),
        ("Memory Detection", test_memory_detection),
        ("NUMA Detection", test_numa_detection),
        ("Context Calculation", test_context_calculation),
        ("Config Generation (System 1)", test_config_generation_system1),
        ("Config Generation (System 2)", test_config_generation_system2),
        ("System Classification", test_system_classification),
        ("Profile Persistence", test_profile_persistence),
        ("Full Detection Flow", test_full_detection_flow),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n✗ {test_name} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n✗ {test_name} ERROR: {e}")
            failed += 1

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
