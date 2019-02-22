import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


torch.manual_seed(1)    # reproducible

# from xx import fun 直接导入模块中某函数，直接fun()就可用。
# 告诉你大法：from xx import * 该模块中所有函数可以直接使用。
# from xx import fun 直接导入模块中某函数，直接fun()就可用。
# 告诉你大法：from xx import * 该模块中所有函数可以直接使用。
# https://www.zhihu.com/question/38857862
# from xx import fun 直接导入模块中某函数，直接fun()就可用。
# 告诉你大法：from xx import * 该模块中所有函数可以直接使用。

EPOCH = 1 # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 64
INPUT_SIZE = 28 # 每個時間點，給的數據是多少個 # 每一行信息包含28個像素點 --> image width
TIME_STEP = 28 # RNN考慮多少個時間點的數據 # 每28步就讀取一行信息 --> image height
LR = 0.01
DOWNLOAD_MNIST = False # 如果還沒有下載過，就設定為True
# Mnist 手写数字

# train_data
train_data = dsets.MNIST(root='./mnist', train=True, transform=transforms.ToTensor(), download=DOWNLOAD_MNIST)
# 保存或者提取位置
# torchvision.datasets有很多可以下載的資料庫
# train=True --> 代表這個train_data是訓練資料
# 我們要把數據轉成tensor的型式
# 转换 PIL.Image or numpy.ndarray成torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
# 要不要download你的數據？如果已經download，就把DOWNLOAD_MNIST設成False

# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True) # 功能-->數據資料-->裝載器
# 把train_data變成train_loader，因為可以使用train_loader來進行一批一批的訓練,提升效率
# 請問為何這邊的train_loader 是用 torch.utils.data.DataLoader，但上一個影片的train_loader卻只用Data.DataLoader呢？有什麼差異嗎？
# CNN那一部裡面有用 import torch.utils.data as Data﻿
# 是一样的，上一节课是在代码上方已经做好了import。﻿

# test_data
test_data = dsets.MNIST(root='./mnist', train=False, transform=transforms.ToTensor())

# test_x = Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)[:2000]/255.
# 數據要用Variable包裝才可以進行反向傳遞計算
# Variable的参数volatile=True和requires_grad=False的功能差不多
# 要把x、y都變成Variable的型式，因為神經網路只能輸入Variable
# 先將variable轉成data的tensor型式，才能由tensor轉成numpy
# Variable的参数volatile=True和requires_grad=False的功能差不多

test_x = test_data.test_data.type(torch.FloatTensor)[:2000]/255.
# test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.
# 如果是照著上面那行，在跑test_output = rnn(test_x)的時候，會出現 rnn input must have 3 dimensions, got 4
# 所以要把unsqueeze給去掉，只留下原本的test_data.test_data，去直接轉成FloatTensor

# squeeze 是减少一个维度, unsqueeze 就是增加一个
# dim=1 --> 加上batch size的維度
# shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
# test_data.test_data --> 從上面自己定義的test_data裡面取出test_data的部分，去提取測試數據
# test_data其實還是0~255的區間

test_y = test_data.test_labels[:2000] # 取最前面2000個，為了範例跑快點
# test_data.test_labels --> 從上面自己定義的test_data裡面取出test_labels的部分，去標記數據
# test y只是用来对比的, 不经过torch加工, 所以不需要像上一行的test_x一樣做成variable

class RNN(nn.Module): # 繼承神經網路裡面，要構成模塊的各項東西
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM( # 直接使用nn.RNN的話，準確率不高，因為很難收斂
            input_size=INPUT_SIZE,
            hidden_size=64, # hidden_size=64是一个lstm  cell 里面有多少神经元
            # 神经元数是它的capacity, 门是lstm 的基本构架。不一样的概念﻿
            num_layers=1, # LSTM的cell只放1層
            batch_first=True
            # 輸入數據有幾個維度，time_step，batch，input_size；如果把batch放在第一個維度，那batch_first就是True
            # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 10) # output的數據要接到fully connected
        # 因為MNIST是手寫數字0~9，所以分10類

    def forward(self, x):
        # (input0, state0) -> LSTM -> (output0, state1);
        # (input1, state1) -> LSTM -> (output1, state2);
        # outputN -> Linear -> prediction. 通过LSTM分析每一时刻的值, 并且将这一时刻和前面时刻的理解合并在一起
        # 生成当前时刻对前面数据的理解或记忆. 传递这种理解给下一时刻分析

        r_out, (h_n, h_c) = self.rnn(x, None) # None 表示 hidden state 会用全0的 state
        # 分线h_n, 主线h_c不是 gate, 而是 lstm 的两种 hidden state.

        # 計算一個input，產生hidden state
        # 然後根據下一個input的時候，除了從圖片裡面的input以外，還要考慮之前產生的hidden state
        # 說明我們已經讀完上面的那些圖片了，知道上面的代表什麼，再加上我們的下一行input的數據，共同產生r_out
        # 然後產生r_out之後，又會產生另一個層次的理解，就是另外一個hidden state，然後一直循環
        # 每一個time_step計算一下input，再計算下一個input...，然後下一個time_step一樣繼續
        # 批數據傳入rnn

        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)

        # 有沒有第一個hidden state？如果沒有，就是None；如果有，就擺進去self.rnn()的第二個引數

        # 之所以使用最後一個output，是因為我們在讀完所有數據之後，才開始做決定
        # 最後一個時刻的時候：hidden stat用不到，但是會r_out會用來做分類的訓練
        # 有28個時間點的output，我們要選取最後一個時間點的output做判斷，因為那是看完完整的圖片
        # 这里 r_out[:, -1, :] 的值也是 h_n 的值
        out = self.out(r_out[:, -1, :]) # r_out shape (batch, time_step, output_size)
        return out

# 建立物件
rnn = RNN()
print(rnn)

# 定義優化器、損失函數
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR) # 去優化剛剛建立的物件，裡面的參數
loss_func = nn.CrossEntropyLoss() # 使用交叉熵當做損失函數
# CrossEntropyLoss，標籤不是onehot型式，而是1、2、3...數字型式，但是程式內部看不到的地方會轉成onehot型式

for epoch in range(EPOCH):
    for step, (x,y) in enumerate(train_loader):
        # (x, y)是train_loader裡面的元素，也就是mini batch
        # step是train step，代表數據更新、優化
        b_x = Variable(x.view(-1, 28, 28)) # reshape x to (batch, time_step, input_size)
        # 在pytorch中, view = reshape
        b_y = Variable(y)
        output = rnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # 數據更新、優化

        if step % 50 == 0: # 每50步來看一次準確率的結果
            test_output = rnn(test_x) # test_output就是預測的結果；samples, time_step, input_size
            pred_y = torch.max(test_output, 1)[1].data.squeeze() # 要從張量轉為numpy，才能餵給matplot

            # 要從張量轉為numpy，才能餵給matplot
            # torch.max(a,1) 返回每一列中最大值的那个元素，且返回索引（返回最大元素在这一行的列索引）
            # troch.max()[1]， 只返回最大值的每个索引

            # torch.max()[0]， 只返回最大值的每个数
            # torch.max(a,0) 返回每一行中最大值的那个元素，且返回索引（返回最大元素在这一列的行索引）
            # https://blog.csdn.net/Z_lbj/article/details/79766690

            # 我们可以利用squeeze()函数将表示向量的数组转换为秩为1的数组，这样利用matplotlib库函数画图时，就可以正常的显示结果了。
            # 那些方程组中真正是干货的方程个数，就是这个方程组对应矩阵的秩。
            # 要先把tensor轉為numpy，才能轉為秩為1，然後才能喂給matplot
            # 两对或以上的方括号形式[[]]，如果直接利用这个数组进行画图可能显示界面为空

            accuracy = (pred_y == test_y).sum().item() / test_y.size()[0]
            # accuracy = (pred_y == test_y).sum().item() / test_y.size()[0]
            # PyTorch不久前刚更新了，本來可以用 accuracy = sum(pred_y == test_y) / test_y.size(0)
            # 這也不能用了：accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))

            print('EPOCH: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
            # 也可以使用loss.item()
            # 之前的 loss 累加为 total_loss +=loss.data[0], 由于现在 loss 为0维张量, 0维检索是没有意义的，所以应该使用 total_loss+=loss.item()
            # https: // zhuanlan.zhihu.com / p / 36307662

# 從test_data印出前10個prediction
test_output = rnn(test_x[:10].view(-1, 28, 28)) # reshape x to (batch, time_step, input_size), 因為batch_first=True
# 在pytorch中, view = reshape
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')

# 为什么将图片改为从左到右依次输入后会效果奇差？﻿
# 原来是我测试集忘了torch.transpose了，只把训练集transpose了一下。。。。﻿
# 意思是原来训练的时候从左到右，测试的时候从上到下？然后发现测试的时候没有transpose？﻿

# 对每一张图像从上到下读入的，请问怎么样操作可以让每一张图像从左到右进行读入？
# 图像就是一个矩阵, 你把矩阵转个方向就好了
