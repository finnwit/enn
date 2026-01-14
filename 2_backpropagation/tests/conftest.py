import sys
import os
import types

# Put this project's root at the front of sys.path so local `src` wins
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Register a minimal `src` package that points to this project's `src` directory.
# This ensures `import src.*` resolves to the local package even if another
# sibling project with a `src` package is present on sys.path.
if 'src' not in sys.modules:
    src_mod = types.ModuleType('src')
    src_mod.__path__ = [os.path.join(ROOT, 'src')]
    sys.modules['src'] = src_mod
