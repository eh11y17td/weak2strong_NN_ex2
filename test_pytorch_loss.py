import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
loss1 = nn.NLLLoss()
loss2 = nn.NLLLoss(ignore_index=0)  # ignore_indexを指定

pred = torch.rand(100, 5)
true = torch.randint(high=5, size=(100,), dtype=torch.long)

print(pred[:20])
print(true[:20])
print(loss1(pred, true))
print(loss1(pred[true > 0], true[true > 0]))
print(loss2(pred, true))