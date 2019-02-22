import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# x是數據，y是標籤；所以x有100筆數據，1行x座標，1行y座標
n_data = torch.ones(100, 2) # 產生元素全是1的2維張量
x0 = torch.normal(2*n_data, 1) # 类型0，第一維象限那團，正态分布随机数
# torch.normal(means, std, out=None)
# 返回一个张量，包含了从指定均值means和标准差std的离散正态分布中抽取的一组随机数
y0 = torch.zeros(100) # 类型0，第一維象限那團，給標籤0
x1 = torch.normal(-2*n_data, 1) # 类型1，第三維象限那團，正态分布随机数
y1 = torch.ones(100) # 类型1，第一維象限那團，給標籤1

# 是将两个张量（tensor）拼接在一起，按维数0（行）拼接
x = torch.cat((x0, x1), 0).type(torch.FloatTensor) # 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor) # 64-bit integer，是torch當中默認的標籤格式
# y0的標籤都是0，y1的標籤都是1，所以是target y

# 把x、y都當做神經網路中的變數去學習
x, y = Variable(x), Variable(y)

#plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# plt.scatter(x.data.numpy()[:, 0], y.data.numpy()[:, 1]
# 如果打成上面這樣 --> too many indices for array，因為y本身就是只有一維而已，沒辦法指定到所有列的'第一行'

# https://blog.csdn.net/qiu931110/article/details/68130199
# cValue = ['r','y','g','b','r','y','g','b','r']
# ax1.scatter(x,y,c=cValue,marker='s') # 第一個點打紅色，第二個點打黃色，第一個點打綠色
# 平常c都是固定一個'r'之類的顏色，如果是序列的話，代表按照scatter的順序，標示上去的顏色會按照給的顏色序列
# 所以這邊c=y.data.numpy()，就是會根據標籤y，來決定這個點打上去的顏色
# s=100 --> 粗點

plt.show()

# class Net(torch.nn.Module):
#     def __init__(self, n_feature, n_hidden, n_output):
#         super(Net, self).__init__()
#         self.hidden = torch.nn.Linear(n_feature, n_hidden)
#         self.predict = torch.nn.Linear(n_hidden, n_output)
#
#     def forward(self, x):
#         x = F.relu(self.hidden(x))
#         # 當要呼叫Net的時候，就要先給定：輸入層、隱藏層、輸出層的節點，所以x的資訊包含了這三層的個數
#         # 反正()裡面都有標明對應的引數順序，所以self.hidden、self.predict都會從x自己去抓對應的需求引數
#         x = self.predict(x)
#         return x
#
# net = Net(2, 10, 2)

net = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    # 跟test7裡面的小寫relu不一樣，那個只是個Python的def relu()，功能而已
    # 而這邊的Relu大寫是個Class，所以會顯示層級出來
    torch.nn.Linear(10, 2)
)
print(net)
# [1, 0] --> 分類標籤是0
# [0, 1] --> 分類標籤是1


# torch.nn.Sequential搭建出來的結果：
# (0): Linear(in_features=2, out_features=10, bias=True)
# (1): ReLU()
# (2): Linear(in_features=10, out_features=2, bias=True)

# class Net(torch.nn.Module) 搭建出來的結果：
# (hidden): Linear(in_features=2, out_features=10, bias=True)
# (predict): Linear(in_features=10, out_features=2, bias=True)










# plt.ion()
# plt.show()
#
# optimizer = torch.optim.SGD(net.parameters(), lr=0.01) # 用SGD去優化net裡面的參數
# # 為了看分類的變化，所以用慢一點的速度做展示
# loss_func = torch.nn.CrossEntropyLoss() # 類別是大寫，資料庫是小寫
# # MSELoss()是用在回歸問題；多分類要用CrossEntropyLoss()
# # 輸出會是softmax [0.1, 0.2, 0.7] --> 各類機率加總要是1
# # [0, 0, 1] --> 標籤誤差； [0.1, 0.2, 0.7] --> 預測誤差
#
# for t in range(100):
#     out = net(x) # 輸入x資訊進去net，會輸出prediction
#     # 回歸的時候是用prediction，但是分類的原始是用out，變成概率後才是用prediction
#     # 原始的分類輸出結果是[-2, -.12, 20]之類的數字，要再用F.softmax(out)轉成概率
#
#     loss = loss_func(out, y) # 預測、Gound Truth，送進loss_func計算誤差
#
#     optimizer.zero_grad() # 清除上一個循環的梯度值，避免累計
#     loss.backward() # 計算誤差反向傳遞，告訴各個節點要有多少gradient
#     optimizer.step() # 然後把這些gradient施加到各個節點去
#
#     if t % 2 == 0: # 每兩步出一張圖
#         plt.cla()
#         # 清除axes，即当前figure 中的活动的axes，但其他axes保持不变
#         # plt.clf() --> 清除当前figure 的所有axes，但是不关闭这个window，所以能继续复用于其他的plot
#
#         prediction = torch.max(F.softmax(out), 1)[1] # troch.max()[1]， 只返回最大值的每个索引
#         # torch.max(a,0) 返回每一行中最大值的那个元素，且返回索引（返回最大元素在这一列的行索引）
#         # torch.max()[0]， 只返回最大值的每个数
#         # https://blog.csdn.net/Z_lbj/article/details/79766690
#         # 那些方程组中真正是干货的方程个数，就是这个方程组对应矩阵的秩。
#         pred_y = prediction.data.numpy().squeeze() # 要先把tensor轉為numpy，才能轉為秩為1，然後才能喂給matplot
#         # 我们可以利用squeeze()函数将表示向量的数组转换为秩为1的数组，这样利用matplotlib库函数画图时，就可以正常的显示结果了。
#         # 两对或以上的方括号形式[[]]，如果直接利用这个数组进行画图可能显示界面为空。
#         target_y = y.data.numpy() # 要把y從張量轉為numpy，才能餵給matplot
#         # y0的標籤都是0，y1的標籤都是1，所以是target y
#         plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
#         # x是數據，y是標籤；所以x有100筆數據，1行x座標，1行y座標
#         accuracy = sum(pred_y == target_y)/200
#         plt.text(0.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
#         plt.pause(0.05)
#
# plt.ioff() # 停止動態畫圖
# plt.show()

