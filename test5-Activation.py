import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt # 如果沒有家pyplot，我的版本會出錯，可是別人的不一定

# fake data
x = torch.linspace(-5, 5, 200) # 使用torch裡面的linspace做切分,這樣就不會是numpy型式，而是tensor型式
# x data (tensor), shape=(100,1)
x = Variable(x) # 將x轉為variable
x_np = x.data.numpy() # 先將variable轉成data的tensor型式，才能由tensor轉成numpy
# torch是不能被plt使用的，所以要轉為numpy的型式

y_relu = F.relu(x).data.numpy() # 將變量x代入，nn.functional裡面的relu函數，然後再把結果轉為numpy型式，才可以丟入plt
y_softplus = F.softplus(x).data.numpy() # 將變量x代入，nn.functional裡面的relu函數，然後再把結果轉為numpy型式，才可以丟入plt

plt.figure(1, figsize=(8,6)) # 使用plt.figure定义一个图像窗口：编号为1；大小为(8, 6)

plt.subplot(221) # 在上面定義的圖像窗口內，設置2*2的子圖，這個子圖放在其中第1個位置
plt.plot(x_np, y_relu, c='red', label='relu') # x、y軸都要是numpy型式，線圖顏色是red
# 如果label打成lable，打錯了，那會跳出超多行的錯誤紅字，看都看不懂
plt.ylim((-1,5)) # y軸範圍是-1~5之間
plt.legend(loc='best') # label要放在子圖的那個位置 --> 'best'，讓它自己選

plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim((-1,5))
plt.legend(loc='best')

plt.show() # 沒有這行的話，畫面會跳出然後自己關掉