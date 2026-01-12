#!/usr/bin/env python3
"""
Web Search Diagnostic Tool

This script tests the web search functionality independently to help diagnose issues.
Run this to see detailed information about what's happening with DuckDuckGo searches.

Usage:
    python test_web_search.py
    python test_web_search.py "your search query"
"""

import sys
import logging

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_duckduckgo_import():
    """Test if duckduckgo-search library is properly installed"""
    print("=" * 60)
    print("1. Testing duckduckgo-search import...")
    print("=" * 60)

    try:
        from duckduckgo_search import DDGS
        import duckduckgo_search
        print(f"✓ duckduckgo-search imported successfully")
        print(f"  Version: {duckduckgo_search.__version__ if hasattr(duckduckgo_search, '__version__') else 'unknown'}")
        print(f"  Location: {duckduckgo_search.__file__}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import duckduckgo-search: {e}")
        print("\nInstall with: pip install duckduckgo-search")
        return False

def test_basic_search(query="Python programming"):
    """Test a basic DuckDuckGo search"""
    print("\n" + "=" * 60)
    print(f"2. Testing basic search: '{query}'")
    print("=" * 60)

    try:
        from duckduckgo_search import DDGS

        print(f"Creating DDGS instance...")
        ddgs = DDGS(timeout=20)

        print(f"Executing search...")
        results = ddgs.text(query, max_results=5)

        print(f"Results type: {type(results)}")

        if results is None:
            print("✗ Search returned None")
            return False

        # Try to iterate
        result_list = []
        for idx, result in enumerate(results):
            print(f"\nResult {idx + 1}:")
            print(f"  Type: {type(result)}")
            print(f"  Keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
            print(f"  Title: {result.get('title', 'N/A')}")
            print(f"  URL: {result.get('href', 'N/A')}")
            print(f"  Snippet: {result.get('body', 'N/A')[:100]}...")
            result_list.append(result)

            if idx >= 4:  # Limit to 5 results
                break

        if result_list:
            print(f"\n✓ Successfully retrieved {len(result_list)} results")
            return True
        else:
            print("\n✗ No results found")
            return False

    except Exception as e:
        print(f"\n✗ Search failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_tools_function(query="Python programming"):
    """Test using the actual tools.py web_search function"""
    print("\n" + "=" * 60)
    print(f"3. Testing tools.py web_search function: '{query}'")
    print("=" * 60)

    try:
        # Import the function from tools
        from tools import web_search

        print(f"Calling web_search('{query}', max_results=3)...")
        result = web_search(query, max_results=3)

        print(f"\nResult:")
        print(f"  Success: {result.get('success')}")
        print(f"  Count: {result.get('count')}")
        print(f"  Error: {result.get('error', 'None')}")

        if result.get('success') and result.get('results'):
            print(f"\nResults found:")
            for idx, res in enumerate(result['results'], 1):
                print(f"\n  Result {idx}:")
                print(f"    Title: {res.get('title', 'N/A')}")
                print(f"    URL: {res.get('url', 'N/A')}")
                snippet = res.get('snippet', '')
                # Remove XML tags for display
                snippet_clean = snippet.replace('<web_search_result>', '').replace('</web_search_result>', '').strip()
                print(f"    Snippet: {snippet_clean[:100]}...")
            print(f"\n✓ tools.py web_search working correctly")
            return True
        else:
            print(f"\n✗ tools.py web_search failed or returned no results")
            return False

    except Exception as e:
        print(f"\n✗ Error testing tools.py: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_network_connectivity():
    """Test basic internet connectivity"""
    print("\n" + "=" * 60)
    print("4. Testing network connectivity...")
    print("=" * 60)

    try:
        import requests

        # Test connection to DuckDuckGo
        print("Attempting to reach duckduckgo.com...")
        response = requests.get("https://duckduckgo.com", timeout=10)

        if response.status_code == 200:
            print(f"✓ Successfully connected to DuckDuckGo (status: {response.status_code})")
            return True
        else:
            print(f"⚠ DuckDuckGo returned status: {response.status_code}")
            return False

    except Exception as e:
        print(f"✗ Network connectivity issue: {type(e).__name__}: {e}")
        return False

def main():
    """Run all diagnostic tests"""
    print("\n" + "=" * 60)
    print("WEB SEARCH DIAGNOSTIC TOOL")
    print("=" * 60)

    # Get query from command line or use default
    query = sys.argv[1] if len(sys.argv) > 1 else "Python programming"

    results = {
        "import": test_duckduckgo_import(),
        "network": test_network_connectivity(),
        "basic_search": False,
        "tools_function": False
    }

    # Only run search tests if import succeeded
    if results["import"]:
        results["basic_search"] = test_basic_search(query)
        results["tools_function"] = test_with_tools_function(query)

    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")

    print("\n" + "=" * 60)

    if all(results.values()):
        print("✓ ALL TESTS PASSED - Web search is working correctly!")
        print("\nIf you're still having issues in the AI server:")
        print("  1. Check server logs (should show 'Web search initiated' messages)")
        print("  2. Verify TOOL_APPROVAL_MODE is True in server_config.py")
        print("  3. Make sure you're approving the web_search tool when prompted")
        print("  4. Check that the model is actually calling the tool (not just answering from knowledge)")
        return 0
    else:
        print("✗ SOME TESTS FAILED - See errors above")
        print("\nCommon issues:")
        print("  1. Library not installed: pip install duckduckgo-search")
        print("  2. Network/firewall blocking DuckDuckGo")
        print("  3. Rate limiting from DuckDuckGo (wait a few minutes)")
        print("  4. Outdated library version: pip install --upgrade duckduckgo-search")
        return 1

if __name__ == "__main__":
    sys.exit(main())
