#!/usr/bin/env python3
"""
Test runner for Spencer AI Backend
"""
import subprocess
import sys
import os

def run_tests():
    """Run all tests with coverage"""
    print("🧪 Running Spencer AI Backend Tests...\n")
    
    # Set Python path
    os.environ['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))
    
    # Test commands
    commands = [
        # Unit tests
        ["pytest", "tests/test_siren_parser.py", "-v", "--tb=short"],
        ["pytest", "tests/test_cms_client.py", "-v", "--tb=short"],
        ["pytest", "tests/test_auth_middleware.py", "-v", "--tb=short"],
        ["pytest", "tests/test_rate_limit_middleware.py", "-v", "--tb=short"],
        ["pytest", "tests/test_base_agent.py", "-v", "--tb=short"],
        
        # Integration tests
        ["pytest", "tests/test_api_integration.py", "-v", "--tb=short"],
        
        # All tests with coverage
        ["pytest", "--cov=.", "--cov-report=term-missing", "--cov-report=html"],
    ]
    
    failed = False
    
    for cmd in commands:
        print(f"\n{'='*60}")
        print(f"Running: {' '.join(cmd)}")
        print(f"{'='*60}\n")
        
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            failed = True
            print(f"\n❌ Test failed: {' '.join(cmd)}")
        else:
            print(f"\n✅ Test passed: {' '.join(cmd)}")
    
    print(f"\n{'='*60}")
    if failed:
        print("❌ Some tests failed!")
        sys.exit(1)
    else:
        print("✅ All tests passed!")
        print("\n📊 Coverage report generated in htmlcov/index.html")
        sys.exit(0)

if __name__ == "__main__":
    run_tests()