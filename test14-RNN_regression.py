import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)

TIME_STEP = 10
INPUT_SIZE = 1 # 設定RNN在那個時間點上的input、output數據都是1
HIDDEN_SIZE = 32
NUM_LAYERS = 1
LR = 0.02

# steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
# x_np = np.sin(steps)
# y_np = np.cos(steps)
# plt.plot(steps, x_np, 'b-', label='input(sin)')
# plt.plot(steps, y_np, 'r-', label='input(cos)')
# plt.legend(loc='best')
# plt.show()

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        # LSTM雖然功能比較強大，但是這邊用RNN就可以滿足了
        self.rnn = nn.RNN( # 建立了類別物件之後，就會內建一個rnn的物件屬性，這個屬性等於是去呼叫torch.nn.RNN的結果
            input_size=INPUT_SIZE, # 1
            hidden_size=HIDDEN_SIZE, # 32
            num_layers=NUM_LAYERS, # 1
            batch_first=True # batch的維度是不是放在第一個
        )
        self.out = nn.Linear(32, 1) # nn套件裡面的fully connected --> hidden輸出32，輸出那一個時間點上的y座標
        # 用Linear的layer，來轉為成1個output

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state) # 因为 hidden state 是连续的, 所以我们要一直传递这一个 state

        # 這層的輸入x、上層輸出的記憶hidden state --> 會輸出這層的r_out、這層輸出的記憶h_state
        # x包含了很多步，比如可能10步的數據一起放進去，但是h_state只有最後一個hidden state
        # RNN的output，有每一步的output，所以每個時間點上的output都放在r_out裡面
        # 所以r_out、h_state的維度是不一樣的

        # x (batch, time_step, input_size) --> 所以batch_first=True
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, output_size) --> (batch, time_step, hidden_size)

        outs = [] # 準備放每一步的r_out，和最後一步的hidden state
        # 保存所有时间点的预测值

        # 動態圖
        for time_step in range(r_out.size(1)): # 对每一个时间点计算 output
            outs.append(self.out(r_out[:, time_step, :])) # 每個時間點輸出一次r_out，進入全連結層，將結果append到outs裡面
        return torch.stack(outs, dim=1), h_state
        # 用torch.stack把他們全部壓到一起，將list轉成tensor的型式
        # h_state的值要傳到下一個廻圈的時候再繼續用

    # def forward(self, x, h_state):
    #     """forward 过程中的对每个时间点求输出还有一招使得计算量比较小的.
    #     不过上面的内容主要是为了呈现 PyTorch 在动态构图上的优势, 所以我用了一个 for loop 来搭建那套输出系统.
    #     這邊介绍一个替换方式. 使用 reshape 的方式整批计算."""
    #     r_out, h_state = self.rnn(x, h_state)
    #     r_out = r_out.view(-1, 32)
    #     outs = self.out(r_out)
    #     return outs.view(-1, 32, TIME_STEP), h_state


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.MSELoss()

plt.figure(1, figsize=(12, 5))
plt.ion() # continuously plot
plt.show()

h_state = None # 變數初始化
# 第一個h_state如果設成None的話，傳到RNN就會自動生成全0的h_state

for step in range(60):
    start, end = step * np.pi, (step+1) * np.pi # 取一小段距離
    # use sin predict cos
    # 取TIME_STEP這麼多個的數據點，然後把這些數據都放到sin、cos裡面去
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps) # float32 for converting torch FloatTensor
    y_np = np.cos(steps)

    x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))
    y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))
    # 從numpy轉成Variable這樣的tensor型式
    # 然後就要把x_np、y_np這兩個sin、cos曲線，包成Variable的型式

    # batch, time_step, input_size
    # 之前的只是1維的數據而已，現在把batch多加了一個維度,為1
    # time_step還是保留原來的這個維度
    # input_size因為也只是一個時間點，那就把它加1個維度

    # 將張量變數x、y傳入RNN
    prediction, h_state = rnn(x, h_state)
    h_state = Variable(h_state.data) # important !!
    # 把h_state包成Variable，然後再傳給下一次
    # 要把 h_state 重新包装一下才能放入下一个 iteration, 不然会报错

    loss = loss_func(prediction, y)    # cross entropy loss
    optimizer.zero_grad()              # clear gradients for this training step
    loss.backward()                    # backpropagation, compute gradients
    optimizer.step()                   # apply gradients

    # plotting
    plt.plot(steps, y_np.flatten(), 'r-') # x軸是step，也就是從0~59步的訓練過程
    # flatten，把數據平坦化變成一維，才可以對照座標畫圖
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-') # 把prediction從tensor轉成numpy才能畫圖
    plt.draw(); plt.pause(0.05)

plt.ioff()
plt.show()

# GitHub代碼
# https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/403_RNN_regressor.py
