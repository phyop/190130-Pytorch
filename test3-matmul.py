import numpy as np
import torch

data = [[1,2], [3,4]]
tensor = torch.FloatTensor(data) # list轉tensor
data = np.array(data) # list轉array

# dot和matmul的区别是，当a或b其中一个是标量的时候，只能用np.dot，用matmul会报错。
print(
    np.matmul(data, data),
    '\n\n',data.dot(data),
    '\n\n',torch.mm(tensor, tensor),
    #'\n\n',tensor.dot(tensor)
    # 在 torch 0.2 版本不能用了，一个解决方案是：
    '\n\n', torch.dot(tensor.view(-1), tensor.view(-1))
)