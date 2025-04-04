import sys
print(f"Python Version: {sys.version}")
print("\nTesting imports...")

try:
    import pandas
    print("✓ pandas available")
except ImportError:
    print("✗ pandas not available")

try:
    import numpy
    print("✓ numpy available")
except ImportError:
    print("✗ numpy not available")

try:
    import matplotlib.pyplot as plt
    print("✓ matplotlib available")
except ImportError:
    print("✗ matplotlib not available")

try:
    import seaborn as sns
    print("✓ seaborn available")
except ImportError:
    print("✗ seaborn not available")

try:
    import scikit_learn
    print("✓ scikit-learn available")
except ImportError:
    try:
        import sklearn
        print("✓ scikit-learn available (as sklearn)")
    except ImportError:
        print("✗ scikit-learn not available")

try:
    import statsmodels
    print("✓ statsmodels available")
except ImportError:
    print("✗ statsmodels not available")

try:
    import tensorflow
    print("✓ tensorflow available")
except ImportError:
    print("✗ tensorflow not available")

print("\nChecking import paths:")
if 'pandas' in sys.modules:
    print(f"pandas path: {sys.modules['pandas'].__path__}")
if 'numpy' in sys.modules:
    print(f"numpy path: {sys.modules['numpy'].__path__}") 