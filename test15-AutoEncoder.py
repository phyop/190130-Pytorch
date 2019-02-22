import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import  matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

torch.manual_seed(1)

# Hyper Parameters
EPOCH = 5
BATCH_SIZE = 64
LR = 0.005
DOWNLOAD_MNIST = False # 因為之前的程式已經下載過了
N_TEST_IMG = 5         # 到时候显示 5张图片看效果

# 下載 MNIST data，並利用DataLoader，來做批訓練
# 這邊只要定義train_data，不用定義test_data，因為是無監督式訓練
# 生成類似於train_data的data，再來跟train_data對比
train_data = torchvision.datasets.MNIST(root='./mnist', train=True,
                                        transform=torchvision.transforms.ToTensor(), # 從MNIST轉成tensor的型式
# Converts a PIL.Image or numpy.ndarray to # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
                                        download=DOWNLOAD_MNIST) # 看要不要從MNIST下載
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# # 第3張圖片的可視化過程 --> [2]
# print(train_data.train_data.size()) # (60000, 28, 28)
# print(train_data.train_labels.size()) # (60000)
# plt.imshow(train_data.train_data[2].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[2])
# plt.show()

class AutoEncoder(nn.Module):
    def __init__(self):
        # 用pytorch的話，要記得有一個initial的過程
        super(AutoEncoder, self).__init__()

        # 壓縮、提取特徵
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128), # 28*28的像素點，放到128個神經元的隱藏層
            nn.Tanh(),
            nn.Linear(128, 64), # 從128個神經元，壓縮成64個
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3),  # 压缩成3个特征, 變成3維的，才能进行3D图像可视化
            # 也可以加上激活函數試試，沒什麼不行，效果好壞很不一定，看經驗
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),  # 激励函数让输出值在 (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)        # x經過encoder被壓縮的結果
        decoded = self.decoder(encoded)  # 壓縮後去被decoder解碼的結果
        return encoded, decoded


autoencoder = AutoEncoder()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
# 如果打成AutoEncoder，會出現parameters() missing 1 required positional argument: 'self'，因為抓到類別，而不是物件
loss_func = nn.MSELoss() # 可以用其他的, 这里图个方便, 用的 MSE

# 初始化畫布
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
# 如果少打了s，打成subplot，會出現ValueError: Illegal argument(s) to subplot: (2, 5)
plt.ion()
plt.show()

view_data = Variable(train_data.train_data[:N_TEST_IMG].view(-1, 28 * 28).type(torch.FloatTensor)/255.)
# 如果打成view(-1, 28, 28)，會出現維度不符的錯誤

for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray')
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())

for epoch in range(EPOCH):
    for step, (x,y) in enumerate(train_loader):
        b_x = Variable(x.view(-1, 28*28))   # 準備進行壓縮的圖片
        # batch x, shape=(batch, 圖片像素)
        b_y = Variable(x.view(-1, 28*28))   # 原數據圖片，用來與壓縮、解壓縮的圖片做對比
        # 因為是無監督，沒有標籤y，所以還是用x
        b_label = Variable(y)               # batch label

        encoded, decoded = autoencoder(b_x)
        # 呼叫autoencoder這個物件之後，因為會自動建立__init__裡面的屬性：self.encoder、self.decoder
        # b_x的引數，只有forward方法可以代入，所以會return encoded, decoded

        loss = loss_func(decoded, b_y)  # 把解壓縮出來的圖片，跟原圖片去做對比
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step % 100 == 0:
            print('EPOCH: ', epoch, '| train loss: %.4f' % loss.item()) # 將loss從tensor轉為numpy的型式
            # 視頻：print('EPOCH: ', epoch, '| train loss: %.4f' % loss.data[0])  # 將loss從tensor轉為numpy的型式

            # plotting decoded image (second row)
            _, decoded_data = autoencoder(view_data)
            # autoencoder物件，一旦傳入引數建立起來，因為只有forward方法可接受引數，所以就會自動回傳encoded、decoded兩個結果
            # 動態5數字辨識，不需要用到encoded_data，所以第1個回傳結果，用_表示
            # 需要用到decoded_data的數據，所以第2個回傳結果，用decoded_data表示

            for i in range(N_TEST_IMG):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(())
                a[1][i].set_yticks(())
            plt.draw()
            plt.pause(0.05)

plt.ioff()
plt.show()


##################################################################################################################


# 2維的5張圖片比較，用到的是decoded_data的數據
# 3維那個類似PCA可視化的圖，用到的是encoded_data的數據
# nn.Linear(12, 3),  # 压缩成3个特征, 變成3維的，才能进行3D图像可视化

view_data = Variable(train_data.train_data[:200].view(-1, 28 * 28).type(torch.FloatTensor)/255.)
# 從train_data這個數據裡面，使用train_data函數，去提取前200個測試數據
# 然後再把這200個測試數據做分批，變成2維數據，(batch，像素) --> (-1, 28 * 28)
# value in range (0, 255) --> (0,1)
# 把這個分批好，並且reshape過，數值控制在0~1之間的數據，用Variable包起來，當做變數準備丟去做訓練

encoded_data, _ = autoencoder(view_data) # 提取压缩的特征值
# autoencoder物件，一旦傳入引數建立起來，因為只有forward方法可接受引數，所以就會自動回傳encoded、decoded兩個結果
# 3D分類，不需要用到decoded_data，所以第2個回傳結果，變數用_表示
# 需要用到encoded_data的數據，所以第1個個回傳結果，用encoded_data表示

fig = plt.figure(2)              # 製作第2個畫布，因為第1個被5數字動態辨識用掉了
ax = Axes3D(fig)                 # 在fig這個畫布上，做3D座標軸
                                 # 然後將數據拆分為3個軸向的數值
X = encoded_data[:, 0].data.numpy()   # 對所有批的第0行tensor數據，轉成numpy，才可以進到matplot
Y = encoded_data[:, 1].data.numpy()   # 對所有批的第1行tensor數據，轉成numpy，才可以進到matplot
Z = encoded_data[:, 2].data.numpy()
values = train_data.train_labels[:200].data.numpy() # 取train_data前200筆的labels，做numpy轉換

# >>>a = [1,2,3]
# >>> b = [4,5,6]
# >>> c = [4,5,6,7,8]
# >>> zipped = zip(a,b)     # 打包为元组的列表
# [(1, 4), (2, 5), (3, 6)]
# >>> zip(a,c)              # 元素个数与最短的列表一致
# [(1, 4), (2, 5), (3, 6)]
# >>> zip(*zipped)          # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式
# [(1, 2, 3), (4, 5, 6)]

# 然後對9類數字分別對做顏色設定，以及3個軸的尺寸設定
for x, y, z ,s in zip(X, Y ,Z , values):       # 因為X~values都是list，所以用x~s各代表他們裡面各自的元素
    c = cm.rainbow(int(255*s/9))               # int(255*s/9) --> 因為數字有0~9，所以分成9類
                                               # 這9類各用不同的隨機顏色做標記，所以使用cm.rainbow
    ax.text(x, y, z, s, backgroundcolor=c)                                  # 0~9字的背景

ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
ax.set_zlim(Z.min(), Z.max())
plt.show()

