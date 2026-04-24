# To verify requirements
# Run this first to confirm your environment is ready.

import torch
import numpy as np

print(f"Pytorch: {torch.__version__}")
print(f"Numpy: {np.__version__}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ MPS (Apple Silicon GPU) available — using it!")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ CUDA available")
else:
    device = torch.device("cpu")
    print("⚠️  Using CPU")

print(f"\nActive device: {device}")

# quick sanity check
x = torch.tensor([1.0, 2.0, 3.0]).to(device)
print(f"Test tensor on {device}: {x}")
print("\n✅ Environment verified.")