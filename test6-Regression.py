# X的范围从[-1,1]变大,如[-5,5]时,会出现梯度爆炸现象(hidden_grad = inf)
# --> 因为 loss 会变大, 而且这时候 learning rate 太大了(0.5)

# 在训练的时候多数情况能拟合得很好，也遇到了这几种情况。请问这问题出在哪里，不稳定因素是什么？
# --> 坏的 initialization 有时候会让激励函数失效. 具体可以参考一下我 pytorch 中的 Batch Normalization 那节的内容

# 如果我有700 Variable，31000行，请问莫烦，要是我放到torch 去训练需要很长时间吗? 我还没有试,不知道是不是需要变成Sparse Matrix?
# --> 我个人的经验来看, pytorch 比 Tensorflow 要快. sparse 按理来说会更快, 建议使用. 其他的我就没有比较过了

# 用Pytroch训练一个数据集，预测值全都是0。请问我应该从哪方面入手查原因呢？
# --> 从这几个方面检查, 降低学习效率, normalization, weights initialization

# jupyter notebook启动网页，好像在上面无法编辑出图片动态可视化的结果
# --> notebook 上运行 matplotlib 好像需要加一句话, 你可以搜搜, 我不太记得了



import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) # unsqueeze把一維數據轉為二維數據，因為torch只會處理二維數據
# 早创造数据的时候我试了一下，不使用unsqueeze来reshape数据也是可以的
y = x.pow(2) + 0.2*torch.rand(x.size()) # x的2次方，然後再加上一些噪音

x, y = Variable(x), Variable(y) # 要把x、y都變成Variable的型式，因為神經網路只能輸入Variable

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

# 自己定義一個網路架構
class Net(torch.nn.Module): # 類別Net繼承自nn.Moudle，才可以在这个 class 中调用nn.Module 的功能
# 類別是大寫，資料庫是小寫
    def __init__(self, n_feature, n_hidden, n_output): # 呼叫Net類別的時候，要key入引數：幾個輸入、幾個隱藏層神經元、幾個輸入
        super(Net, self).__init__() # 繼承自Net的__init__
        self.hidden = torch.nn.Linear(n_feature, n_hidden) # 從輸入層，到隱藏層神經元
        self.predict = torch.nn.Linear(n_hidden, n_output) # 從隱藏層神經元，到輸出層
        # 只要呼叫Net類別，就要key入引數，所以當然要先定義這些引數代表的意義
        # 比如：hidden代表從輸入層到輸出層的連結數，predict代表從隱藏層到輸出層的連結數

    def forward(self, x):
        x = F.relu(self.hidden(x)) # 把輸入x先丟去hidden層，再把輸出的結果，丟去nn.functional裡面的relu函數
        x = self.predict(x) # 然後繼續再把relu的輸出丟去predict層
        return x # 完成從輸入到輸出層的步驟了

net = Net(1, 10, 1)
print(net)

# 把matplotlib變成一個時時打印的過程
plt.ion() # on --> 打印開始
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5) # optimizer會去優化 'net.parameters'
loss_func = torch.nn.MSELoss() # mean square

for t in range(100):
    prediction = net(x) # 就是喂数据, 输出预测嘛.
    # prediction = net.forward(x) # 同上面那行

    loss = loss_func(prediction, y) # 真實值要放在後面
    # loss是一個variable，因為是從torch.nn出來的

    optimizer.zero_grad() # 讓每一次廻圈開始，把net的所有參數梯度都先降為0 (i.e. net.parameters)
    # 因為每一次計算loss以後，梯度都會保留在optimizer裡面
    # 在call loss.backward()和 optimizer.step()之前需要把上一轮得到的梯度清零﻿
    # 因为这是新一轮的更新, gradient 和上一次的不一样了. 如果不进行这一步, gradient 可能会被累加(我还没试过..)
    loss.backward() # 然後才開始這一次的反向傳遞過程
    # loss是一個variable，經過反向傳遞給神經網路當中的每一個節點,計算出這些節點的梯度
    optimizer.step() # 計算出各節點的梯度後，用optimizer來優化這些梯度

    if t % 5 == 0: # 每五步就打印一次
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), '-r', lw=5) # 拟合出来的是一条水平线，不是完全拟合 --> 调小学习率，增加迭代次数
        # plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0] # 會出現format有問題 --> Use tensor.item() to convert
        # 如果是#到下面這行，而不是上面那行，那plt.pause(0.1)這行就會出錯
        # 然後藉由下面一路#下去追查到結尾，就會得到unexpected EOF while parsing，才發現因為是偵測到上面那行還沒有斷句
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)


    plt.ioff()  # off --> 打印結束
    # plt.show()

# print(list(net.parameters())) # 在训练完成后查看回归模型的系数(weights and bias)