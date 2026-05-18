import os
import random
import numpy as np

# 这个最好在 CUDA / cuBLAS 真正使用前设置
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "42"

import torch

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch.use_deterministic_algorithms(True, warn_only=True)

torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

print("[Deterministic] Enabled: seed=42, cudnn.benchmark=False, TF32=False")