import torch
import numpy as np
import math


X = torch.rand(2, 2, dtype=torch.float)

print(X.mean(-1, keepdim=True))