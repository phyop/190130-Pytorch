import torch
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

N_SAMPLES = 20
N_HIDDEN = 300

# training data
x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
y = x + 0.3*torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))

# test data
test_x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
test_y = test_x + 0.3*torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))

# show data
plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.5, label='train')
plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.5, label='test')
plt.legend(loc='upper left')
plt.ylim((-2.5, 2.5))
plt.show()

# 過擬合 --> 數據量太少，或是神經網路太強大(太深、太多神經元)
# 所以我們現在只有20個數據，卻有300個神經元去擬合它們 --> 形成過擬合現象

net_overfitting = torch.nn.Sequential(
    torch.nn.Linear(1, N_HIDDEN),         # 一個神經元輸入，輸出到N_HIDDEN個神經元
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, N_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, 1),         # 輸出層
)

# 在這邊，是把dropout添加在Linear和ReLU之間
# 也是有人是放在ReLU跟Linear之間。反正就自己試看看
net_dropped = torch.nn.Sequential(
    torch.nn.Linear(1, N_HIDDEN),
    torch.nn.Dropout(0.5),  # drop 50% of the neuron
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, N_HIDDEN),
    torch.nn.Dropout(0.5),  # 如果是0.2，就代表每次隨機drop掉20%的神經元
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN, 1),
)

# 看看兩種神經網路的結構差異
print(net_overfitting)  # net architecture
print(net_dropped)

optimizer_ofit = torch.optim.Adam(net_overfitting.parameters(), lr=0.01)
optimizer_drop = torch.optim.Adam(net_dropped.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()

plt.ion()   # something about plotting

for t in range(500):                        # 訓練500次
    pred_ofit = net_overfitting(x)          # 每次把x正向傳遞
    pred_drop = net_dropped(x)              # 經過兩種不同的神經網路傳遞
    loss_ofit = loss_func(pred_ofit, y)     # 然後各別計算兩種網路的誤差量 # train模式
    loss_drop = loss_func(pred_drop, y)     # train模式

    optimizer_ofit.zero_grad()              # train模式
    optimizer_drop.zero_grad()              # train模式
    loss_ofit.backward()
    loss_drop.backward()
    optimizer_ofit.step()
    optimizer_drop.step()

    if t % 10 == 0:
        # 做預測的時候，先把dropout取消，看看原本會長怎麼樣 --> net_dropped.eval()
        # 做訓練的時候，才把dropout打開，看看成果會是怎麼樣 --> net_dropped.train()
        # change to eval mode in order to fix drop out effect
        net_overfitting.eval()              # 預測模式
        # overfitting本來就沒有drop功能，所以使用evaluation來取消drop功能，其實跟原本一樣，也就是沒有作用
        net_dropped.eval()                  # 預測模式
        # 把原本的net_dropped這個網路，裡面的dropout功能取消掉

        # 如果把上面兩行的eval給#掉，也把最下面的train也#掉
        # --> 預測的時候，就沒有把drop的功能取消掉，所以matplot的預測圖就也會顯示過擬合

        # plotting
        plt.cla()
        test_pred_ofit = net_overfitting(test_x)    # 拿測試數據來跑看看測試結果
        test_pred_drop = net_dropped(test_x)        # 比較兩種網路的測試結果
        plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.3, label='train')
        plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.3, label='test')
        plt.plot(test_x.data.numpy(), test_pred_ofit.data.numpy(), 'r-', lw=3, label='overfitting')
        plt.plot(test_x.data.numpy(), test_pred_drop.data.numpy(), 'b--', lw=3, label='dropout(50%)')
        plt.text(0, -1.2, 'overfitting loss=%.4f' % loss_func(test_pred_ofit, test_y).data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.text(0, -1.5, 'dropout loss=%.4f' % loss_func(test_pred_drop, test_y).data.numpy(), fontdict={'size': 20, 'color': 'blue'})
        plt.legend(loc='upper left'); plt.ylim((-2.5, 2.5));plt.pause(0.1)

        # change back to train mode
        net_overfitting.train()             # 從eval模式轉回train模式
        net_dropped.train()                 # 從eval模式轉回train模式

plt.ioff()
plt.show()