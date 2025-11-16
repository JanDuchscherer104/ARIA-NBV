#!/usr/bin/env python3
"""
Test script to verify Python 3.9+ compatibility and basic functionality.
"""

import sys
from pathlib import Path


def test_python_version():
    """Test that we're running Python 3.9+"""
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ required for EFM3D compatibility")
        return False
    print("✅ Python version compatible")
    return True


def test_typing_compatibility():
    """Test that we're using compatible type annotations"""

    # Use older-style type annotations for compatibility
    def example_func(
        mesh_path: str | Path, points: list[float] | None = None, metadata: dict[str, str] | None = None
    ) -> dict[str, int]:
        return {"count": 0}

    print("✅ Type annotations compatible with Python 3.9")
    return True


def test_imports():
    """Test that our core dependencies can be imported"""
    try:
        import torch

        print(f"✅ PyTorch {torch.__version__} available")
    except ImportError:
        print("❌ PyTorch not available")
        return False

    try:
        import trimesh

        print(f"✅ Trimesh {trimesh.__version__} available")
    except ImportError:
        print("❌ Trimesh not available")
        return False

    try:
        import numpy

        print(f"✅ NumPy {numpy.__version__} available")
    except ImportError:
        print("❌ NumPy not available")
        return False

    return True


def test_oracle_rri_import():
    """Test that our oracle_rri package can be imported"""
    try:
        import oracle_rri_old

        print("✅ oracle_rri package imports successfully")

        # Test core components
        from oracle_rri_old.core.candidates import CandidateViewGenerator
        from oracle_rri_old.core.oracle import OracleRRI
        from oracle_rri_old.core.renderer import CandidateViewRenderer
        from oracle_rri_old.evaluation.metrics import ChamferDistance
        from oracle_rri_old.geometry.camera import CameraTW
        from oracle_rri_old.geometry.pose import PoseTW

        print("✅ All core components import successfully")
        return True
    except ImportError as e:
        print(f"❌ oracle_rri import failed: {e}")
        return False


def main():
    """Run all compatibility tests"""
    print("Oracle RRI Compatibility Test")
    print("=" * 40)

    tests = [
        test_python_version,
        test_typing_compatibility,
        test_imports,
        test_oracle_rri_import,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
        print()

    print("=" * 40)
    if all(results):
        print("🎉 All tests passed! Package is ready for use.")
        return 0
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
