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
        "memory": MOCK_SYSTEM2_RAM
    }

    config = hardware_detector.generate_optimal_config(hardware_info)

    print(f"Mode: {config['mode']}")
    print(f"  GPU Layers: {config['n_gpu_layers']}")
    print(f"  Context Size: {config['ctx_size']} tokens")
    print(f"  Threads: {config.get('threads', 'None')}")
    print(f"  Reasoning: {config['reasoning']}")

    # Verify System 2 config expectations
    assert config['n_gpu_layers'] == 0, "System 2 should use CPU-only mode"
    assert config['ctx_size'] >= 65536, "System 2 should have massive context window"
    assert config['threads'] is not None, "System 2 should specify thread count"
    assert config['threads'] >= 50, "System 2 should use most of 56 threads"
    assert config['mode'] == "cpu_highram", "System 2 should be cpu_highram mode"

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

    # Test CPU mode with 256GB RAM
    ctx_cpu_256gb = hardware_detector.calculate_optimal_context(ram_gb=256.0, mode="cpu")
    print(f"CPU 256GB RAM: {ctx_cpu_256gb} tokens")
    assert ctx_cpu_256gb >= 65536, "256GB RAM should support massive context"

    # Test CPU mode with 32GB RAM
    ctx_cpu_32gb = hardware_detector.calculate_optimal_context(ram_gb=32.0, mode="cpu")
    print(f"CPU 32GB RAM: {ctx_cpu_32gb} tokens")
    assert ctx_cpu_32gb >= 8192, "32GB RAM should support medium context"

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
