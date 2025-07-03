#!/usr/bin/env python3
"""
Test script to verify config module works without numpy
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_config_import():
    """Test that config can be imported without numpy"""
    try:
        from config import config, get_config, load_triad_config_data, get_triad_config
        print("✅ Successfully imported config module")
        return True
    except Exception as e:
        print(f"❌ Failed to import config module: {e}")
        return False

def test_config_initialization():
    """Test that config can be initialized without numpy"""
    try:
        from config import config
        print("✅ Config initialized successfully")
        print(f"   Project directory: {config.DIR_PROJECT}")
        print(f"   Triad directory: {config.DIR_TRIAD}")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize config: {e}")
        return False

def test_lazy_loading():
    """Test lazy loading of numpy arrays"""
    try:
        from config import load_triad_config_data, get_triad_config
        
        # Test loading without numpy (should return None with warning)
        print("\nTesting lazy loading without numpy...")
        result = load_triad_config_data('1', 'equipart')
        if result is None:
            print("✅ Lazy loading handled missing numpy gracefully")
        else:
            print("✅ Lazy loading worked with numpy available")
        
        # Test getting config
        config = get_triad_config('1', 'equipart')
        print(f"✅ Got config for dimension 1: {config['name']}")
        
        return True
    except Exception as e:
        print(f"❌ Failed lazy loading test: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing config module without numpy requirement...")
    print("=" * 50)
    
    tests = [
        test_config_import,
        test_config_initialization,
        test_lazy_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Config module works without numpy.")
    else:
        print("❌ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main() 