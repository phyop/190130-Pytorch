import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1,2],[3,4]]) # 轉為浮點張量
variable = Variable(tensor, requires_grad = True) # 要不要吧Variable涉及到反向傳播去
# 如果要涉及，那就會計算Variable這個節點當中的gradient

print(tensor.type)
print(variable.type)

t_out = torch.mean(tensor*tensor) # x^2
v_out = torch.mean(variable*variable)

# tensor不能反向傳播，Variable可以反向傳播

print(t_out)
print(v_out)

v_out.backward() # 我們要算v_out的反向傳遞
print(variable.grad) # 但是看variable的梯度
# 因為v_out的根本變數是variable
# v_out = 1/4*(2*variable) = 1/2*variable

print(variable) # variable的形式

print(variable.data) # tensor的形式

print(variable.data.numpy()) # 然後藉由tensor的形式，才能轉為numpy的形式