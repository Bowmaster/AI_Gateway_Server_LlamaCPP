"""
test_server.py - Basic test script for AI Lab Server
Tests communication between ai_server -> llama_manager -> llama-server
"""

import requests
import time
import sys

SERVER_URL = "http://localhost:8080"

def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def test_health():
    """Test 1: Health check"""
    print_section("TEST 1: Health Check")
    
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Server is responding")
            print(f"  Status: {data['status']}")
            print(f"  Model: {data['model_key']} ({data['model_name']})")
            print(f"  Device: {data['device']}")
            print(f"  Model Loaded: {data['model_loaded']}")
            print(f"  GPU Layers: {data['n_gpu_layers']}")
            print(f"  Tools: {'Enabled' if data['tools_enabled'] else 'Disabled'}")
            
            return data['model_loaded']
        else:
            print(f"✗ Server returned error: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to server at {SERVER_URL}")
        print("  Make sure ai_server.py is running")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_models_list():
    """Test 2: List available models"""
    print_section("TEST 2: Models List")
    
    try:
        response = requests.get(f"{SERVER_URL}/models", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Retrieved models list")
            print(f"  Current Model: {data['current_model_key']}")
            print(f"\n  Available Models:")
            
            for model in data['models']:
                status = "✓ CURRENT" if model['is_current'] else ("★" if model['recommended'] else " ")
                exists = "✓" if model['exists'] else "✗ NOT FOUND"
                print(f"    [{status}] {model['key']:<20} {exists}")
                print(f"        {model['description']}")
            
            return True
        else:
            print(f"✗ Failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_simple_chat():
    """Test 3: Simple chat without tools"""
    print_section("TEST 3: Simple Chat (No Tools)")
    
    payload = {
        "messages": [
            {"role": "user", "content": "Hello! Please respond with just: 'Hello, I am working correctly.'"}
        ],
        "enable_tools": False,
        "max_tokens": 50
    }
    
    try:
        print("Sending chat request...")
        response = requests.post(f"{SERVER_URL}/chat", json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Chat successful")
            print(f"\n  Response: {data['response']}")
            print(f"\n  Stats:")
            print(f"    Tokens: {data['tokens_input']} in → {data['tokens_generated']} out → {data['tokens_total']} total")
            print(f"    Time: {data['generation_time']:.2f}s @ {data['tokens_per_second']:.1f} tok/s")
            print(f"    Device: {data['device']}")
            
            return True
        else:
            print(f"✗ Failed: {response.status_code}")
            print(f"  Error: {response.json().get('detail', 'Unknown error')}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"✗ Request timed out")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_chat_with_tools():
    """Test 4: Chat with tool calling"""
    print_section("TEST 4: Chat with Tools")
    
    payload = {
        "messages": [
            {"role": "user", "content": "What is the IP address of google.com?"}
        ],
        "enable_tools": True,
        "max_tokens": 256
    }
    
    try:
        print("Sending chat request with tools enabled...")
        response = requests.post(f"{SERVER_URL}/chat", json=payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Chat successful")
            print(f"\n  Response: {data['response']}")
            
            if data.get('tools_used'):
                print(f"\n  Tools Used: {', '.join(data['tools_used'])}")
            else:
                print(f"\n  Note: No tools were used (model may not have called tools)")
            
            print(f"\n  Stats:")
            print(f"    Tokens: {data['tokens_input']} in → {data['tokens_generated']} out → {data['tokens_total']} total")
            print(f"    Time: {data['generation_time']:.2f}s @ {data['tokens_per_second']:.1f} tok/s")
            
            return True
        else:
            print(f"✗ Failed: {response.status_code}")
            print(f"  Error: {response.json().get('detail', 'Unknown error')}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"✗ Request timed out")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_system_prompt():
    """Test 5: Custom system prompt"""
    print_section("TEST 5: Custom System Prompt")
    
    # Set custom system prompt
    try:
        response = requests.post(
            f"{SERVER_URL}/command",
            json={"command": "system", "value": "You are a pirate. Always respond like a pirate."},
            timeout=5
        )
        
        if response.status_code == 200:
            print("✓ System prompt set")
        else:
            print(f"✗ Failed to set system prompt")
            return False
    except Exception as e:
        print(f"✗ Error setting system prompt: {e}")
        return False
    
    # Test with new system prompt
    payload = {
        "messages": [
            {"role": "user", "content": "Hello!"}
        ],
        "enable_tools": False,
        "max_tokens": 50
    }
    
    try:
        print("Sending chat request with pirate system prompt...")
        response = requests.post(f"{SERVER_URL}/chat", json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Chat successful")
            print(f"\n  Response: {data['response']}")
            
            # Reset system prompt
            requests.post(
                f"{SERVER_URL}/command",
                json={"command": "system", "value": "reset"},
                timeout=5
            )
            print(f"\n  (System prompt reset to default)")

            return True
        else:
            print(f"✗ Failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_streaming_simple():
    """Test 6: Streaming chat without tools"""
    print_section("TEST 6: Streaming Chat (No Tools)")

    payload = {
        "messages": [
            {"role": "user", "content": "Count from 1 to 5, one number per line."}
        ],
        "enable_tools": False,
        "max_tokens": 100
    }

    try:
        print("Sending streaming request...")
        accumulated = ""
        token_count = 0
        metadata = {}

        with requests.post(
            f"{SERVER_URL}/chat/stream",
            json=payload,
            stream=True,
            timeout=60
        ) as response:
            if response.status_code != 200:
                print(f"  ✗ Failed: {response.status_code}")
                return False

            for line in response.iter_lines():
                if not line:
                    continue

                line = line.decode('utf-8')
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break

                    try:
                        import json
                        chunk = json.loads(data_str)

                        # Check for metadata
                        if chunk.get("type") == "stream_end":
                            metadata = chunk
                            continue

                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta:
                            token = delta["content"]
                            accumulated += token
                            token_count += 1
                            print(token, end="", flush=True)
                    except json.JSONDecodeError:
                        pass

        print(f"\n\n  ✓ Received {token_count} tokens")
        print(f"  Response preview: {accumulated[:80]}...")
        if metadata:
            print(f"  Generation time: {metadata.get('generation_time', 'N/A')}s")
            print(f"  Device: {metadata.get('device', 'N/A')}")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_streaming_with_tools():
    """Test 7: Streaming with tools (hybrid mode)"""
    print_section("TEST 7: Streaming with Tools (Hybrid)")

    payload = {
        "messages": [
            {"role": "user", "content": "What is the IP address of example.com?"}
        ],
        "enable_tools": True,
        "max_tokens": 200
    }

    try:
        print("Sending streaming request with tools enabled...")
        accumulated = ""
        tools_used = None
        metadata = {}

        with requests.post(
            f"{SERVER_URL}/chat/stream",
            json=payload,
            stream=True,
            timeout=120
        ) as response:
            if response.status_code != 200:
                print(f"  ✗ Failed: {response.status_code}")
                return False

            for line in response.iter_lines():
                if not line:
                    continue

                line = line.decode('utf-8')
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break

                    try:
                        import json
                        chunk = json.loads(data_str)

                        if chunk.get("type") == "stream_end":
                            metadata = chunk
                            tools_used = chunk.get("tools_used")
                            continue

                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta:
                            accumulated += delta["content"]
                            print(delta["content"], end="", flush=True)
                    except json.JSONDecodeError:
                        pass

        print(f"\n\n  ✓ Streaming complete")
        print(f"  Response preview: {accumulated[:100]}...")
        if tools_used:
            print(f"  Tools used: {', '.join(tools_used)}")
        else:
            print(f"  Note: No tools were used")
        if metadata:
            print(f"  Generation time: {metadata.get('generation_time', 'N/A')}s")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("  AI Lab Server Test Suite")
    print("=" * 60)
    print(f"  Target: {SERVER_URL}")
    print("=" * 60)
    
    results = []
    
    # Test 1: Health check
    result = test_health()
    results.append(("Health Check", result))
    
    if not result:
        print("\n✗ Server is not responding. Make sure ai_server.py is running.")
        sys.exit(1)
    
    # Wait a moment between tests
    time.sleep(1)
    
    # Test 2: Models list
    result = test_models_list()
    results.append(("Models List", result))
    time.sleep(1)
    
    # Test 3: Simple chat
    result = test_simple_chat()
    results.append(("Simple Chat", result))
    time.sleep(1)
    
    # Test 4: Chat with tools
    result = test_chat_with_tools()
    results.append(("Chat with Tools", result))
    time.sleep(1)
    
    # Test 5: System prompt
    result = test_system_prompt()
    results.append(("System Prompt", result))
    time.sleep(1)

    # Test 6: Streaming (no tools)
    result = test_streaming_simple()
    results.append(("Streaming (No Tools)", result))
    time.sleep(1)

    # Test 7: Streaming with tools (hybrid)
    result = test_streaming_with_tools()
    results.append(("Streaming (Hybrid)", result))

    # Summary
    print_section("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}  {test_name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All tests passed! Server is working correctly.")
        sys.exit(0)
    else:
        print(f"\n  ⚠️  {total - passed} test(s) failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
