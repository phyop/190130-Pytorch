# https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/406_GAN.py

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


torch.manual_seed(1) # torch用的亂數
np.random.seed(1)    # numpy用的亂數

BATCH_SIZE = 64
LR_G = 0.0001           # generator 的學習率
LR_D = 0.0001           # discriminator 的學習率
N_IDEAS = 5             # 新手畫家的隨機發想，有5個靈感。因為作画的时候需要有一些灵感
# 一幅画需要有一些规格, 我们将这幅画的画笔数定义一下,
ART_COMPONENTS = 15     # N_COMPONENTS 就是一条一元二次曲线(这幅画画)上的点个数.
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])
# 为了进行批训练, 我们将一整批话的点都规定一下(PAINT_POINTS).
# PAINT_POINTS是從-1~1的線段，有15個點
# 要對 BATCH 裡面所有的資料，都各別做同樣的15個點的線段切割，所以用for...in range(BATCH的資料數)
# 因為 BATCH 裡面的元素之後的for迴圈沒有要再使用了，所以用 _ 來表示不重要，可丟棄

# # 畫家的漂亮作品 - 要在紅、藍色曲線的區間
# plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='up_bound')
# plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='down_bound')
# plt.legend(loc='upper right')
# plt.show()

# 这里我们生成一些著名画家的画 (batch 条不同的一元二次方程曲线).
def artist_works():
    # 生成隨機，但是均勻的弧線

    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis] # 批量生成隨機數
    # 在[1, 2)間生成一個隨機數，如果size=5：
    # array([[1.43064969],
    #        [1.28363844],
    #        [1.31212298],
    #        [1.93143556],
    #        [1.730056]])
    paintings = a * np.power(PAINT_POINTS, 2) + (a-1)           # 以numpy制定畫家畫圖的曲線函數
    # 畫的圖就是二次曲線，所以paintings = a * x^2 + (a-1)
    # 最終生成在藍色、紅色曲線間的區間
    # paintings設為曲線，也就是一元二次方程式
    # PAINT_POINTS是從-1~1的線段，有15個點，每一個點就用來產生一元二次函數
    # a就是一元二次函數當中的一個參數

    # import random --> random.uniform(x, y) --> 返回一个浮点数，在[x, y) 范围内
    # 注意：uniform()是不能直接访问的，需要导入 random 模块，然后通过 random 静态对象调用该方法。
    # a = random.uniform(10, 100) --> 会出现10-100之间的实数，但是有小数
    # a = int(random.uniform(10, 100)) --> 出来的随机数就都是整数了

    # [:, np.newaxis] --> 把它加一個維度
    # 有一個一維陣列x1
    # x1 = np.array([10, 20, 30], float) --> x1.shape = (3,)
    # [10. 20. 30.]
    # x2 = x1[:, np.newaxis] --> x2.shape = (3, 1)
    # [[10.]
    #  [20.]
    #  [30.]]
    # x3 = x1[np.newaxis, :] --> x3.shape = (1, 3)
    # [[10. 20. 30.]]

    # 然後再把從numpy定義好的畫圖曲線，轉換成torch的型式，然後再轉成float，最後用Variable包起來
    paintings = torch.from_numpy(paintings).float()             # 從numpy轉成可訓練的float、Variable格式
    return Variable(paintings)

# Generator，新手画家用前向傳遞生成畫
# G網路拿著N_IDEAS,來創造出一幅畫(ART_COMPONENTS)
G = nn.Sequential(
    nn.Linear(N_IDEAS, 128),         # N_IDEAS = 新手畫家的隨機發想，有5個靈感。用這5個靈感做出一幅畫
    nn.ReLU(),
    nn.Linear(128, ART_COMPONENTS)   # ART_COMPONENTS就是生成的線段，有15個點
) # 用隨機的5個靈感，生成由15個點做成的線段

# Discriminator(新手鉴赏家)
# 可能是從新手畫家接收到其輸出，也可能是從artist_works()的paintings接收到的
D = nn.Sequential(
    nn.Linear(ART_COMPONENTS, 128),
    nn.ReLU(),
    nn.Linear(128, 1),               # 看接受到的畫，是不是著名畫家的畫
    nn.Sigmoid()                     # 所以我們要把它轉換成一個百分比的機率型式
)

# 定義完生成、判別器這兩個神經網路之後，我們就要分別定義其優化器
opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

plt.ion()

# 建構好神經網路、優化器以後，就可以開始學習
# G 首先会有些灵感, G_ideas 就会拿到这些随机灵感画画 (可以是正态分布的随机数)
# 接着我们拿着著名画家的画和 G 的画, 让 D 来判定这两批画作是著名画家画的概率.
for step in range(10000):
    artist_paintings = artist_works()                    # 創造出著名畫家的畫
    G_ideas = Variable(torch.randn(BATCH_SIZE, N_IDEAS)) # 新手畫家的idea，是由隨機數的靈感來開始畫的
    # 一個batch裡面N_IDEAS個數據，縱坐標的個數是BATCH_SIZE，橫坐標是N_IDEAS
    G_paintings = G(G_ideas)                             # 新手畫家的畫；G --> 新手画家用前向傳遞生成畫

    # 比較著名、新手畫家，各自為名著的概率
    prob_artist0 = D(artist_paintings)                   # artist0 --> 著名畫家的作品，看看有多少概率是來自著名畫家
    prob_artist1 = D(G_paintings)                        # artist1 --> 新手畫家的作品，看看有多少概率是來自著名畫家

    # 把誤差反向傳遞回去
    D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1-prob_artist1))
    # 希望增加prob_artist0的機率，也就是著名畫家的作品，被判別為確實來自著名畫家
    # 盡量減小prob_artist1，這些畫是新手畫家的畫的作品，卻被誤認為來自著名畫家
    # 因為優化是去minimize誤差，而不是最大化，所以才加上個負號

    # G就是想要增加自己雖然是新手畫家，但是希望可以被當成著名畫家，所以要增加artist1的概率
    # 也就是增加1-prob_artist1這個的概率
    G_loss = torch.mean(torch.log(1-prob_artist1))

    # torch 中提升参数的形式是最小化误差, 那我们把最大化 score 转换成最小化 loss, 在两个 score 的合的地方加一个符号就好.
    # 而 G 的提升就是要减小 D 猜测 G 生成数据的正确率, 也就是减小 D_score1.


    opt_D.zero_grad() # 判別優化器先初始化歸零
    D_loss.backward(retain_graph=True) # 然後開始向後傳遞。
    opt_D.step()      # 更新網路
    # retain_variables=True --> 要保留參數給下面的G_loss.backward()用
    # retain_graph 这个参数是为了再次使用计算图纸

    opt_G.zero_grad() # 判別優化器先初始化歸零
    G_loss.backward() # 然後開始向後傳遞
    opt_G.step()      # 更新網路

    # 可視化，每50步秀一次變化
    if step % 50 == 0:
        plt.cla()
        # plt.cla() --> 清除axes，即当前figure 中的活动的axes，但其他axes保持不变
        # plt.clf() --> 清除当前figure 的所有axes，但是不关闭这个window，所以能继续复用于其他的plot
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting',)
        # PAINT_POINTS --> 要對 BATCH 裡面所有的資料，在-1~1之間都各別做同樣的15個點的線段切割。當做橫坐標
        # 新手畫家畫出來的圖，轉成numpy當做縱坐標

        # # 畫家的漂亮作品 - 要在紅、藍色曲線的區間
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(),
                 fontdict={'size': 13})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
        plt.ylim((0, 3));
        plt.legend(loc='upper right', fontsize=10);
        plt.draw();
        plt.pause(0.01) # 延长显示的秒數
        # 在plt.show()之前使用plt.draw()，在绝大多数情况下都是多余的.
        # 您可能需要它的唯一时间是您正在进行一些不涉及使用pyplot函数的非常奇怪的修改.

plt.ioff()
plt.show()


