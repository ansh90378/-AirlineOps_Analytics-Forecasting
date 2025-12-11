import subprocess
import os

# Ensure Kaggle API is available
try:
    import kaggle
except ImportError:
    raise ModuleNotFoundError("Install Kaggle first: pip install kaggle")

DATASET = "usdot/flight-delays"   # Kaggle dataset name
DEST = "."                        # download to current folder

print("Downloading Kaggle dataset...")
subprocess.run([
    "kaggle", "datasets", "download",
    "-d", DATASET,
    "-p", DEST,
    "--unzip"
])

print("✅ Download complete. Files saved in:", os.path.abspath(DEST))
