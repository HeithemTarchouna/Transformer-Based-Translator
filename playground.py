import torch
import numpy as np
import math


X = torch.rand(2, 2, dtype=torch.float)

Y = X.clone()

print(Y)

#print(torch.ones(1,3,3).triu(1).int())

print(torch.tensor([1]).repeat(3))
