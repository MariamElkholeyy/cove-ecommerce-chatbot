import sys
from pathlib import Path
from importlib.metadata import version

print("Python:", sys.version.split()[0])
print("Interpreter:", sys.executable)
assert sys.prefix != sys.base_prefix, "Select this project's .venv notebook kernel."
for package in ["ipykernel", "numpy", "pandas", "scikit-learn", "matplotlib"]:
    print(f"{package}: {version(package)}")

import numpy as np
import pandas as pd
import sklearn
import matplotlib

example = pd.DataFrame({"message": ["Hello", "Where is my order?"], "example_type": ["greeting", "support request"]})
print(example.to_string(index=False))
assert np.array([1, 2, 3]).sum() == 6
print("\nSETUP CHECK PASSED")
