# https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/501_why_torch_dynamic_graph.py

import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
INPUT_SIZE = 1          # rnn input size / image width
LR = 0.02               # learning rate


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        # 繼承自nn.Module裡面RNN的屬性

        self.rnn = nn.RNN(           # 建立物件後，利用這個物件本身(也就是self)去呼叫名為rnn屬性的時候，就等同是呼叫torch.nn.RNN
            input_size=1,
            hidden_size=32,          # rnn hidden unit
            num_layers=1,            # number of rnn layer
            batch_first=True,        # 對input & output來說，要把batch size放第一順位 (batch, time_step, input_size)
        )
        # x進去nn.RNN之後，會經過1層hidden layer，裡面包含32個隱藏神經元，然後再經由self.out的全連接層輸出成1個結果
        self.out = nn.Linear(32, 1)  # 利用這個物件本身去呼叫名為out屬性的時候，就等同是做32-->1的線性全連結層

    def forward(self, x, h_state):
        # batch_first=True --> 對input & output來說，要把batch size放第一順位
        # x       (batch   , time_step, input_size)
        # r_out   (batch   , time_step, output_size)
        # h_state (n_layers, batch    , hidden_size)  --> hidden state的batch放第2個

        r_out, h_state = self.rnn(x, h_state)
        # torch.nn.RNN    --> 輸入兩引數，會輸出兩引數
        # 輸入x, h_state進去self.rnn，會輸出r_out, h_state
        # 因為steps是1~3隨機，所以輸入不同的time_step，會有不同長度時間的輸出
        # 然後再把r_out進行下面的加工、合併、輸出

        outs = []
        # 先初始化一個list，準備放東西進去
        # this is where you can find torch is dynamic

        for time_step in range(r_out.size(1)):
            # calculate output for each time step
            # r_out -->  (batch, time_step, output_size) -->  (0, 0, output_size), (0, 1, output_size), (0, 2, output_size)...
            # torch的size，就像是numpy的shape，所以 r_out.size(1) --> 就是指有多少個 time_step

            outs.append(self.out(r_out[:, time_step, :]))
            # 把r_out丟進去(32, 1)全連接層，再把結果append到outs去
            # 把r_out[:, time_step, :]，也就是以每一個time_step為單位，送去輸入到self.out這個全連接層
            # r_out --> (batch, time_step, output_size)

        return torch.stack(outs, dim=1), h_state


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.MSELoss()
# 如果少了()，當最下面的loss在呼叫loss_func的時候會報錯
# bool value of Tensor with more than one value is ambiguous

h_state = None # 一開始傳入的hidden state為None，可能因為不是遷移學習吧

plt.figure(1, figsize=(12, 5)) # 畫布的大小
plt.ion()                      # continuously plot

# 用x的sin曲線，來預測y的cos的曲線
step = 0
for step in range(60):
    # start, end = step * np.pi, (step+1) * np.pi
    # steps = np.linspace(start, end, 10, dtype=np.float32)
    # print(len(steps)) # 經過幾個step喂給RNN
    #
    # # TensorFlow中，batch=None 是因為batch_size常有變化
    # # 但是time_step也常有變化啊，那如果在TensorFlow中，batch、time_step都設為None，就會報錯
    # # 因為在TensorFlow中，不能有兩個隨機變化的維度
    #
    # # 像上面的steps，linspace 裡面的數據點是10個，可是我們可能會要有其他的個數，比如20個、30個
    # # 也就是print(len(steps))會每run都印出10，代表每次都是經過10個step喂給RNN
    # # 所以我們可以用numpy的random，來代替end裡面(step+1)中的“+1”，以及steps裡面linspace的“10”

    ###################################################################################################################

    # 讓steps數，在1、2、3這3個數中隨機選取
    dynamic_steps = np.random.randint(1, 4) # randn()：在()範圍內，取隨機整數
    step += dynamic_steps

    start, end = step * np.pi, (step + dynamic_steps) * np.pi
    steps = np.linspace(start, end, 10 * dynamic_steps, dtype=np.float32)

    print('steps:', len(steps))

    x_np = np.sin(steps)
    y_np = np.cos(steps)

    # 用Variable包起來，才能跑optim優化
    x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))
    y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))
    # [:, np.newaxis] --> 把它增加維度
    # x = (batch, time_step, input_size)
    # batch默認是None

    prediction, h_state = rnn(x, h_state)
    # 因為r_out, h_state = self.rnn(x, h_state)，所以prediction = r_out
    h_state = Variable(h_state.data)
    # github上面寫的是：h_state = h_state.data
    # Variable裡面要包的是numpy格式，所以要先把h_state從tensor轉過去numpy
    # h_state經由Variable進行優化傳遞後，還是h_state

    loss = loss_func(prediction, y) # 前面預測，後面真實
    optimizer.zero_grad()           # clear gradients for this training step
    loss.backward()                 # backpropagation, compute gradients
    optimizer.step()                # apply gradients

    # plotting
    # y_np是cos圖，也就是真實數據，是numpy型式
    # prediction是預測，是tensor型式
    plt.plot(steps, y_np.flatten(), 'r-')                    # 把y數據攤平，才能當做y座標畫圖
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-') # 要轉成numpy型式才能丟進matplot
    plt.draw()                                               # 某些畫圖有問題的情況，可以在show之前加上draw()指令
    plt.pause(0.01)

plt.ioff()
plt.show()

#####################################################################################################################

        # 《torch.stack》
        # 把兩個list合在一起 --> stack 或是 cat
        # stack會增加一個維度

        # a = torch.IntTensor([[1,2,3], [11,22,33]])； a.size() --> torch.Size([2, 3])
        # b = torch.IntTensor([[4,5,6], [44,55,66]])； b.size() --> torch.Size([2, 3])
        # dim=0 --> a不指定維度，所以取全部的元素
        # dim=1 --> a指定一個維度下的元素
        # dim=2 --> a指定2個維度下的元素

        # dim = 0时，相當於不變
        # dim = 1时，相當於做轉置
        # dim = 2时，相當於取list內部元素做轉置

        # c = [a, b]
        # d = [[a[0], b[0]],   [a[1], b[1]]]
        # e = [[[a[0][0], b[0][0]],   [a[0][1], b[0][1]],   [a[0][2], b[0][2]]],
        #      [[a[1][0], b[1][0]],   [a[1][1], b[0][1]],   [a[1][2], b[1][2]]]]

        # c = torch.stack([a, b], dim=0)
        # c.size() --> torch.Size([2, 2, 3])
        # 2個2列3行的東西放在一起，所以就是(2,(2,3))
        # tensor([[[1, 2, 3],
        #          [11, 22, 33]],
        #         [[4, 5, 6],
        #          [44, 55, 66]]], dtype=torch.int32)

        # d = torch.stack([a, b], dim=1)
        # d.size() --> torch.Size([2, 2, 3])
        # tensor([[[ 1,  2,  3],
        #          [ 4,  5,  6]],
        #
        #         [[11, 22, 33],
        #          [44, 55, 66]]], dtype=torch.int32)

        # e = torch.stack([a, b], dim=2)
        # c.size() --> torch.Size([2, 3, 2])
        # tensor([[[1, 4],
        #          [2, 5],
        #          [3, 6]],
        #
        #         [[11, 44],
        #          [22, 55],
        #          [33, 66]]], dtype=torch.int32)

#####################################################################################################################
        # 《np.stack》

        # list裡面的元素是int的時候，
        # 很直觀的；
        # hstack — > 全部打掉，水平一排
        # vstack — > 不變，跟原本一樣

        # list裡面的元素是list的時候，
        # 相當於做轉置；
        # hstack — > 垂直變水平 — > -y轉到 + x
        # vstack — > 水平變垂直 — > +x轉到 - y

        # axis = 0 — > 不變；
        # axis = 1 — > 轉置，行變列
        #
        # (4, 2, 3) — >
        # 最內層的int元素有3個；
        # 有2個這樣的list；
        # 總共4組；

        # 從：
        # np.shape(a) — > (2, 3)
        # np.shape(b) — > (2, 3)
        # np.shape(c) — > (2, 3)
        # np.shape(d) — > (2, 3)
        #
        # 變成：
        # e = np.stack((a, b, c, d), axis=0)
        # np.shape(e) — > (4, 2, 3)
        #
        # e = np.stack((a, b, c, d), axis=1)
        # np.shape(e) — > (2, 4, 3)
        #
        # 所以就算增加了新的維度

#####################################################################################################################