import torch
import numpy as np

data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data) # 轉為浮點張量

print(
    np.abs(data),
    '\n',torch.abs(tensor)
)
