from __future__ import annotations

import platform

import torch
import transformers
import tokenizers
import peft
import trl

EXPECTED = {
    "transformers": "4.57.1",
    "tokenizers": "0.22.1",
    "peft": "0.17.1",
    "trl": "0.27.2",
}

print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Tokenizers:", tokenizers.__version__)
print("PEFT:", peft.__version__)
print("TRL:", trl.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))

actual = {
    "transformers": transformers.__version__,
    "tokenizers": tokenizers.__version__,
    "peft": peft.__version__,
    "trl": trl.__version__,
}
for name, expected in EXPECTED.items():
    if actual[name] != expected:
        print(f"WARNING: reference release used {name}=={expected}; found {actual[name]}")
