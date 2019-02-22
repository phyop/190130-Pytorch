import torch
import torch.nn as nn
import torchvision # 含有訓練數據庫
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.utils.data as Data

EPOCH = 1 # 練習而已，為了省時間
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False # 下載好了以後，就改成False

train_data = torchvision.datasets.MNIST( # 如果datasets少打s，就會出現module 'torchvision' has no attribute 'dataset'
    root='./mnist',
    train=True, # 如果False，那就會給test data
    # train有六萬個，test有1萬2
    transform=torchvision.transforms.ToTensor(), # 把下載的數據，轉為Tensor的型式(原本可能是nd.array的數據)
    # RGB 三個顏色，總共算是一個維度
    download=DOWNLOAD_MNIST# 如果已經download，就把DOWNLOAD_MNIST設為false
    # 如果transform或download的最後，沒有打逗點的話，那等號左邊就不會顯示顏色，因為代表還沒結束
)

# print(train_data.train_data.size()) # 自己定義的XX，去呼叫計算train_data 大小的函數
# print(train_data.train_labels.size()) # 自己定義的XX，去呼叫計算train_labels 大小的函數
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray') # 把train_data轉為numpy型式，才能使用imshow函數
# plt.title('%i' % train_data.train_labels[0])
# plt.show()


train_loder = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False) # train=False，代表不是training data
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.test_labels[:2000] # 取最前面2000個，範例為了快速
# shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
# dim=1 --> 加上batch size的維度
# 數據要用Variable包裝才可以進行反向傳遞計算
# test_data.test_data --> 從test_data這個數據裡面，使用test_data函數，去提取測試數據
# test_data其實還是0~255的區間

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d( # (1, 28, 28) --> 1就是channel維度，28*28是圖片寬、高
                in_channels=1, # 進來的圖片是灰階，所以深度只有1
                out_channels=16, # 輸出的深度是16，16個filters同時進行操作，提取16個特徵到下一層去
                kernel_size=5,
                stride=1,
                padding=2 # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ), # output shape (16, 28, 28) # 輸出是16個channel
            # 過濾器是有高度、寬度、深度，是3維的；深度是特徵
            nn.ReLU(), # nn下面直接接的話，都是類別，所以都是大寫；如果是接functional，在那之後就是小寫
            nn.MaxPool2d(kernel_size=2) # (16, 14, 14)
            # 經過Conv2d之後，因為截取圖片特徵多，就會變深
            # 有1d，2d，3d
        )
        self.conv2 = nn.Sequential( # (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2), # (32, 14, 14)
            # Conv1出來的是16層，所以這邊的輸入是16層，然後我們把它加工成32層
            # kenel_size、stride、padding維持5、1、2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # (32, 7, 7) --> 這是3維度的數據，我們要在fully connected那邊攤平
        )
        self.out = nn.Linear(32 * 7 * 7, 10) # 展平成一個二維的數據，分成10個分類

    def forward(self, x):
        x = self.conv1(x) # conv裡面包括Conv2d、ReLU、MaxPool2d
        x = self.conv2(x) # (batch, 32, 7, 7)，因為是跑mini batch的訓練，所以這邊把batch的維度考慮進來了
        # 自己在CNN類別裡面定義的屬性，這個屬性包含完整的網路架構
        x = x.view(x.size(0), -1) # x.size(0) --> 把batch的維度保留
        # -1的意思就是，把(32, 7, 7)全部展平，放到一起，變成(32 * 7 * 7)
        # 所以經過x.view展平的操作，就變成(batch, 32 * 7 * 7)
        output = self.out(x) # (32 * 7 * 7, 10) --> 32 * 7 * 7 的輸入，10 的輸出
        # 然後才可以放到output layer
        return output




# for (b_x, b_y) in train_loader:
# 只有像(b_x, b_y)這樣才是0~1
# 現在還沒壓縮，我們要進行壓縮

cnn = CNN() # 建立一個CNN類別的物件，名為cnn
print(cnn)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR) # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss() # the target label is not one-hotted

for epoch in range(EPOCH): # 每一個epoch都要跑一輪
    for step, (x, y) in enumerate(train_loder): # 每一個epoch裡面有很多個batch要訓練
        # step --> training step
        # train_loder 裡面有這些參數：dataset、batch_size、shuffle、num_workers
        # mini batch x、y
        b_x = Variable(x) # 把原始資料x、y都要用Variable包裝起來
        b_y = Variable(y)

        output = cnn(b_x)
        loss = loss_func(output, b_y) # b_y是target label；output是prediction
        optimizer.zero_grad() # 對這個training step，要進行梯度清除
        loss.backward() # 清除了以後，就可以開始跑反向傳遞
        optimizer.step() # 然後開始優化各個參數

        if step % 50 == 0: # 每50個step，看一次訓練效果
            # 利用test_data來計算accuracy
            test_output = cnn(test_x) # test_output就是預測的結果
            pred_y = torch.max(test_output, 1)[1].data.squeeze() # 要從張量轉為numpy，才能餵給matplot
            # troch.max()[1]， 只返回最大值的每个索引
            # torch.max()[0]， 只返回最大值的每个数
            # torch.max(a,0) 返回每一行中最大值的那个元素，且返回索引（返回最大元素在这一列的行索引）
            # torch.max(a,1) 返回每一列中最大值的那个元素，且返回索引（返回最大元素在这一行的列索引）
            # https://blog.csdn.net/Z_lbj/article/details/79766690

            # 我们可以利用squeeze()函数将表示向量的数组转换为秩为1的数组，这样利用matplotlib库函数画图时，就可以正常的显示结果了。
            # 那些方程组中真正是干货的方程个数，就是这个方程组对应矩阵的秩。
            # 要先把tensor轉為numpy，才能轉為秩為1，然後才能喂給matplot
            # 两对或以上的方括号形式[[]]，如果直接利用这个数组进行画图可能显示界面为空。

            accuracy = (pred_y == test_y).sum().item() / test_y.size()[0]
            # PyTorch不久前刚更新了，本來可以用 accuracy = sum(pred_y == test_y) / test_y.size(0)
            # 這也不能用了：accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('EPOCH: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
            # 也可以使用loss.item()
            # 之前的 loss 累加为 total_loss +=loss.data[0], 由于现在 loss 为0维张量, 0维检索是没有意义的，所以应该使用 total_loss+=loss.item()
            # https: // zhuanlan.zhihu.com / p / 36307662

# 從test_data印出前10個prediction
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy, 'real number')

# github代碼
# https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/401_CNN.py

# 請問為何代碼：test_y = test_data.test_labels[:2000] 不需要像上一行的test_x一樣做成variable呢？
# test y 只是用来对比的, 不经过 torch 加工, 所以就没有变﻿

# 不好意思還有個問題，我上網查了有關squeeze跟unsqueeze，還是不太知道這兩個的意義到底是什麼﻿
# squeeze 是减少一个维度, unsqueeze 就是增加一个﻿