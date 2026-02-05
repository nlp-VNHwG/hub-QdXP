# check_env.py
try:
    import jieba

    print(f"✓ jieba installed successfully, version: {jieba.__version__}")
except ImportError as e:
    print(f"✗ jieba not installed: {e}")


try:
    import sklearn

    print(f"✓ sklearn installed successfully, version: sklearn installed")
except ImportError as e:
    print(f"✗ sklearn not installed: {e}")


try:
    import torch

    print(f"✓ torch installed successfully, version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"✗ torch not installed: {e}")
