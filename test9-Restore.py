import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(2) # torch.manual_seed(1)的时候拟合线是一条直线，改成2才能出来最后的效果
# manual_seed 给PyTorch里的随机数生成器指定一个随机种子，可以保证每次生成的随机数是从随机种子开始取值的
# 就是说每次调用 torch.rand() 的结果都是可以复现的
# 我觉得这是个坑。用同样的种子来运行CPU的程序应该可以重复过去的实验
# 然而，如果用了GPU，某些运算操作（如卷积）是有随机因素的
# 原因超出我的知识范畴，貌似是为了运算优化，只保证结果小数点后N位小数是准确的。如此以来就产生了蝴蝶效应
# 即使用了同样的种子初始化，训练出来的模型也是不同的

# fake data
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1) # shape=(100, 1)
y = x.pow(2) + 0.2*torch.rand(x.size()) # y=x^2+雜訊，x.size()=(100, 1)
x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False) # 把x、y設定為變量

# 在训练时如果想要固定网络的底层，那么可以令这部分网络对应子图的参数requires_grad为False。
# 这样，在反向过程中就不会计算这些参数对应的梯度
# Variable的参数volatile=True和requires_grad=False的功能差不多

# torch.rand(2, 3)
# 0.0836 0.6151 0.6958
# 0.6998 0.2560 0.0139
# unsqueeze()在dim维插入一个维度为1的维，例如原来x是n×m维的，torch.unqueeze(x,0)这返回1×n×m的tensor
# queeze() 函数功能：去除size为1的维度，包括行和列。当维度大于等于2时，squeeze()无作用。
# 其中squeeze(0)代表若第一维度值为1则去除第一维度，squeeze(1)代表若第二维度值为1则去除第二维度。

# x = torch.zeros(4,3,2)
# tensor([[[0., 0.], # 最小的括號裡面有2個元素
#          [0., 0.],
#          [0., 0.]], # 次小的括號有3組最小括號
#         [[0., 0.],
#          [0., 0.],
#          [0., 0.]],
#         [[0., 0.],
#          [0., 0.],
#          [0., 0.]],
#         [[0., 0.],
#          [0., 0.],
#          [0., 0.]]]) # 再次小的括號有4組次小括號

def save():
    # save net
    # 之後要用net2、net3來提取出來
    net1 = torch.nn.Sequential( # 利用Sequantial來建立一系列流程
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()
    # 都要先定義網路、優化器、損失函數

    for t in range(100): # 迭代100次
        # 一樣要先定義網路、優化器、損失
        prediction = net1(x) # 在最開始定義了輸入數據x，也在上面定義了net1的網路架構，所以運算之後就是輸出prediction
        loss = loss_func(prediction, y) # 後面放ground truth
        optimizer.zero_grad()
        # Before the backward pass, use the optimizer object to zero all of the gradients for the variables it will update
        loss.backward() # 將損失函數向後傳遞
        optimizer.step() # 只有用了optimizer.step()，模型才会更新

    torch.save(net1, 'net.pkl') # entire
    # 要傳進去的參數，要保存的名字
    torch.save(net1.state_dict(), 'net_params.pkl') # 只保存网络中的参数 (速度快, 占内存少)

    plt.figure(1, figsize=(10, 3)) # 這個figure的代稱為1
    plt.subplot(131) # 1列3行的子圖®，其中的第1個
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5) # 先顏色，再線條的款式(虛線)


def restore_net():
    net2 = torch.load('net.pkl')
    prediction = net2(x)

    plt.figure(1, figsize=(10, 3))  # 這個figure的代稱為1
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)


def restore_params(): #因為要截取net1的參數過來，所以要建立一樣的網路
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    net3.load_state_dict(torch.load('net_params.pkl'))
    # 如果上面那行結尾少了)，下面這行就會出錯，然後就算把下面這行#掉，下下行還是會出錯

    prediction = net3(x)

    plt.figure(1, figsize=(10, 3))  # 這個figure的代稱為1
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)



save() # 保存 net1 (1. 整个网络, 2. 只有参数)
restore_net() # 提取整个网络
restore_params() # 提取网络参数, 复制到新网络

plt.show() # 如果在三個def各自plt.show()，那就會分成3張圖
# 如果plt.figure(1, figsize=(10, 3))沒有在每個def裡面都寫，比如只寫在第3個呼叫的地方
# 那變成呼叫第一個的時候沒有這樣的figsize定義，就會變成3個def的畫布大小不同，造成無法按照想要的畫布大小呈現
# 如果不要在每個def裡面都寫，也可以直接另外寫一個def來定義畫布，不然也可以像plt.show()一樣，寫在主文(save())的前、後。