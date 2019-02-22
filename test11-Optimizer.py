import torch
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(1)    # reproducible

# 用全大寫來表示超參數
LR = 0.01 # Learning Rate
BATCH_SIZE = 32
EPOCH = 12

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1) # unsqueeze把一維數據轉為二維數據，因為torch只會處理二維數據
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))
# 不打星號也沒差，就如同預設的 "def forward(self, *input):" 一樣，那個*是代表還沒輸入，要改
# x.size() = [1000, 1]，所以是做成一個[1000, 1]的矩陣，裡面的元素都是0
# 然後再經由normal，來把所有元素變正態分佈。如此變成一個與x維度相同的矩陣

# torch.normal(means, std, out=None)
# 返回一个张量，包含了从指定均值means和标准差std的离散正态分布中抽取的一组随机数

plt.scatter(x.numpy(), y.numpy())
plt.show()

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

class Net(torch.nn.Module): # 繼承類別
    # __init__要先定義一些屬性
    def __init__(self): # 要輸入的參數 --> 一定跟自己的類別有關係
        super(Net, self).__init__() # 繼承自己類別總不會有錯；參數一定跟自己有關係
        self.hidden = torch.nn.Linear(1, 20) # hidden layer
        self.predict = torch.nn.Linear(20, 1) # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x)) # 輸入x，經由Linear通過隱藏層
        # torch.nn.functional裡面有的是relu，小寫，只是一個神經網路的一個小函數功能，所以放在funcitonal裡面
        # torch.nn.ReLU()，大寫，所以是一個大類別
        x = self.predict(x) # 然後再送去輸出層
        return x

# 利用Net類別，來建立四個網路
net_SGD = Net()
net_Momentum = Net()
net_RMSprop = Net()
net_Adam = Net()
nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

opt_SGD      = torch.optim.SGD    (net_SGD.parameters()     , lr=LR) # 利用optim裡面的SGD優化器，去優化net_SGD網路裡面的參數
opt_Momentum = torch.optim.SGD    (net_Momentum.parameters(), lr=LR, momentum=0.8) # 也是SGD，只是加了momentum的屬性
opt_RMSprop  = torch.optim.RMSprop(net_RMSprop.parameters() , lr=LR, alpha=0.9) # RMSprop是alpha，包含1個參數
opt_Adam     = torch.optim.Adam   (net_Adam.parameters()    , lr=LR, betas=(0.9, 0.99)) # Adam是betas,包含2個參數
optimizers   = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

loss_func = torch.nn.MSELoss()
losses_his  = [[], [], [], []] # recode losster(x.numpy(), y.numpy()) # 前幾個範例都是使用 y.data.numpy
plt.show()



for epoch in range(EPOCH):
    print('Epoch: ', epoch)
    for step, (b_x, b_y) in enumerate(loader):
    # 只有用了optimizer.step()，模型才会更新
    #     b_x = Variable(batch_x) # 把數據用Variable包起來
    #     b_y = Variable(batch_y)
        # x、y、torch_dataset、loader 都是tensor的型式，不是Variable的型式
     # 视频里面有做一步 ：b_x=Variable(batch_x)，b_y=Variable(batch_y)
    # 但这一步好像没有在本页的讲解跟源码里出现，我试了做或不做这一步结果好像也没什么影响，

        # 不同的神經網路，1個1個拿出來訓練
        for net, opt, l_his in zip(nets, optimizers, losses_his):
            # nets, optimizers, losses_his，這3個東西都是list型式
            # net, opt, l_his，這3個東西，分別代表nets, optimizers, losses_his的list內元素
            output = net(b_x) # 用nets這個list裡面所有的元素，依序跑下去
            loss = loss_func(output, b_y) # loss_func 同一用MSELoss()去計算誤差
            opt.zero_grad() # 將優化器裡面的梯度清除，以免累計到下一個epoch
            loss.backward() # 確定梯度清除以後，才開始這次epoch的反向傳遞計算
            # 根據loss來進行反向傳播
            opt.step() # 對優化器模型進行更新
            l_his.append(loss.data.numpy()) # 把loss轉成data型式，依序append進losses_his裡面(也就是l_his，這個代表)



labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
for i, l_his in enumerate(losses_his):
# losses_his  = [[], [], [], []] # recode loss
# 'SGD', 'Momentum', 'RMSprop', 'Adam']的loss，依序append進losses_his裡面
    plt.plot(l_his, label=[i]) # l_his是losses_his裡面的元素
plt.legend(loc='best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim((0, 0.2))
plt.show()