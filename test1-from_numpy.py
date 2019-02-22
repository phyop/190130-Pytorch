import torch
import numpy as np

np_data = np.arange(6).reshape((2,3))
torch_data = torch.from_numpy(np_data) # from numpy轉tensor
tensor2array = torch_data.numpy() # 轉numpy

print(
    '\nnp_data\n', np_data,'\n',type(np_data),
    '\n\ntorch_data\n', torch_data,'\n',type(torch_data),
    '\n\ntensor2array\n', tensor2array,'\n',type(tensor2array),
)