import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import gym


BATCH_SIZE          = 32
LR                  = 0.01
EPSION              = 0.1   # greedy policy，最优选择动作百分比
GAMMA               = 0.9   # reward discount，奖励递减参数
TARGET_REPLACE_ITER = 100   # target update frequency，Q现实网络的更新频率
MEMORY_CAPACITY     = 2000  # 记忆库大小
env                 = gym.make('CartPole-v0')         # 立杆子游戏
#  WARN: gym.spaces.Box autodetected dtype as <class 'numpy.float32'>.
#  警告信息不是我们的错，而是Gym内部的一个小小的不一致，并不影响结果。
env                 = env.unwrapped                   # 把遊戲解壓縮出來
# unwrapped沒有括號
N_ACTIONS           = env.action_space.n              # 遊戲能做的动作
# n沒有括號
N_STATES            = env.observation_space.shape[0]  # 遊戲能获取的环境信息数


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 10)   # 輸入觀測狀態，也就是遊戲能获取的环境信息数，輸出到有10個神經元的隱藏層
        # fully connected全連結層
        self.fc1.weight.data.normal_(0, 0.1) # 機器學習中，隨機生成的學習起始值
        # initialization，均值μ=0，方差σ=0.1的正态分布。
        self.out = nn.Linear(10, N_ACTIONS)  # 輸出動作 N_ACTIONS 的價值有多少
        self.out.weight.data.normal_(0, 0.1) # weight.data.normal_(mean, std)

    def forward(self, x):
        # 輸入N_STATES經過向前傳遞網路，輸出動作值的價值
        # 再把價值的動作，丟到DQN網路，讓DQN去選取動作
        x = self.fc1(x) # 呼叫__init__的屬性代入
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

        # nn.functional裡面放一些小函數，所以relu是小寫
        # ReLU是類別，是層的概念，在nn裡面，functional的外面


class DQN(object):
    def __init__(self):
        # 建立 target net 和 eval net 还有 memory
        # 现实网络 (Target Net)；估计网络 (Eval Net)
        # 简化的 DQN 体系是这样, 我们有两个 net, 有选动作机制, 有存经历机制, 有学习机制
        # 定義神經網路的類別，才需要super(Net, self).__init__()

        self.eval_net, self.target_net = Net(), Net()
        # DQN是Q-Learning的一種方法，但是會有兩個神經網路
        # Q-target、Q-eval，實際上是2個等同的神經網路，但是參數設置上有不同
        # 要時不時的把eval參數，轉換到target參數，讓它有延遲的更新效果

        self.learn_step_counter = 0   # 看學習到多少步了，用于 target 更新计时
        self.memory_counter = 0       # 记忆库位置的一個counter
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2)) # 初始化全0的记忆库，2000列的記憶容量*4行(s、s_、a、r)
        # 看store_transition，知道存了哪些東西：(N_STATES * 2 --> 狀態、下一個狀態)、(+ 2 --> 動作、回饋)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr= LR) # torch 的优化器
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        # 根據環境的觀測值，決定採取的動作 --> 要滑動到哪邊，才可以維持桿子的平衡
        # x就是輸入的觀測值，用Variable包起來，去跑神經網路優化
        # squeeze 是减少一个维度, unsqueeze 就是增加一个
        # 在位置0的地方，增加一個批訓練的維度
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))

        # 这里只输入一个 sample
        # 隨機選取動作的概率 --> EPSILON，如果這個概率小於我們的隨機數呢，就採取greedy的行為
        # greedy --> 兩個動作從forward方法回傳的價值，選取高的那個
        # 现实网络 (Target Net)；估计网络 (Eval Net)
        if np.random.uniform() < EPSION: # 选最优动作
            # your epsilon-greedy is wrong...The code works fine, but the definition of epsilon greedy is in on the other way,
            # in your case, your epsilon should be 0.1, if the random number is greater than 0.1,
            # you should choose the greedy version( the network generate one),
            # otherwise it should random generate the action.

            actions_value = self.eval_net.forward(x)
            # self.eval_net = Net()，也就是將網路做前向傳遞
            # forward --> 輸入N_STATES經過向前傳遞網路，輸出動作值的價值
            action = torch.max(actions_value, 1)[1].data.numpy()[0] # 選取action裡面最大的那個價值，return the argmax
            # 返回每一列中最大值的那个元素的索引，並轉成numpy
            # 由于 pytorch 的版本变化，.data.numpy()[0, 0] 可能需要改成.data.numpy()[0]

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

        # 这里只输入一个 sample
        else: # 如果不是greedy的話，那動作就從N_ACTIONS動作中隨機選取一個
            action = np.random.randint(0, N_ACTIONS) # 從動作中選擇一個動作
        return action
        # return 看最後我真正選取的action到底哪一個

    # 看下面強化學習的過程，其中的for廻圈，選取完action之後，還要存儲記憶
    # a = dqn.choose_action(s)  # DQN根據現在的狀態，來採取行為; a代表action
    # s_, r, done, info = env.step(a)  # 根據採取的行為，環境給的反饋；s_代表行為後的state
    # dqn.store_transition(s, a, r, s_)  # DQN會存儲這些資訊來學習：之前的狀態、施加的動作、環境給的reward、環境導引去的下一個狀態



    def store_transition(self, s, a, r, s_): # state, action, reward, state_bar
        # 選取完action之後，還要存儲記憶
        # 存储记忆，DQN的記憶庫，學習的過程就是從記憶庫裡面提取記憶，然後進行強化學習的方法，用Q-learning的方法去學習它
        transition = np.hstack((s, [a, r], s_)) # 記憶捆在一起之後，存儲在相對應的位置

        index = self.memory_counter % MEMORY_CAPACITY
        # 如果memory_counter超過記憶容量上限，就重新開始索引
        # 也就是，如果记忆库满了, 就覆盖老数据
        self.memory_counter += 1
        # 所以要記錄一下：存一個，就把儲存的counter加1次


    def learn(self):
        # target net 参数更新
        # 檢測要不要把更新target_net (從eval_net)
        # 学习记忆库中的记忆
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0 : # 100步更新一次，如果到了那個步數就更新
            # TARGET_REPLACE_ITER = 100 --> 现实网络的更新频率
            # self.learn_step_counter --> 用于 target 更新计时，看學習到多少步了
            self.target_net.load_state_dict(self.eval_net.load_state_dict())
            # target_net的更新 --> target_net中的所有參數，從eval_net複製過來
            # eval_net在每一次learn的時候都在更新，而target_net是每100步更新一下
        self.learn_step_counter += 1
        # 莫煩大大，你的 self.learn_step_counter 是不是忘記更新值了，它永遠都是0
        # 还正是, 哈哈哈, 我补上了.没写那句话, 神经网络居然还能学, 好厉害

        # 提取一批記憶，使用批訓練更新
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE) # 從記憶庫裡面，隨機抽取BATCH_SIZE=32個記憶
        b_memory = self.memory[sample_index, :] # batch memory = 從2000列的記憶容量中，隨機選取32個記憶，對(s、s_、a、r)都做
        # 记忆库 self.memory --> 2000列的記憶容量*4行(s、s_、a、r)

        # 然後把s, a, r, s_打包成Variable的樣子，就可以丟去神經網路做學習了
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES])) # 所有列的，前面N_STATES這麼多行，不包含第N_STATES行
        b_a = Variable(torch.FloatTensor(b_memory[:, N_STATES:N_STATES + 1].astype(np.int64))) # 所有列的，第N_STATES行，和它的右邊一行
        # np.int32类型在新版本里不能直接转化为longtensor，需要把astype（int）改为astype（np.int64）即可
        # 因为astype（int）相当于astype（np.int32）
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])) # 第N_STATES+1行，和它的右邊一行
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:])) # 從最右邊數來，前面N_STATES這麼多行

        # deep Q-learning的學習過程
        # 計算q_evaluation(預測值)、q_next、q_target(真實值)
        # 要計算q_eval、q_target的差異
        # 現在的狀態丟到eval_net，就會生成所有動作的價值，再從裡面選取我們當初施加動作的價值，就是q_eval
        # 再把這個當初施加動作的價值q_eval，去跟q_target相減
        # q_target就等於下一步q的價值，加上當初獲得的獎勵*GAMMA
        # 經過神經網路taret_net(b_s_)分析出來的下一步q，是對每一個動作的q_next，但是我們要選擇當中最大q值的那個動作
        # 要注意的是，反向傳遞之後，會更新網路的參數，但是我們不要q_target被反向傳遞更新到
        # 因為q_target是要等到q_eval學了100步之後才要來更新，而不是每次反向傳遞就更新一次


        q_eval = self.eval_net(b_s).gather(1, b_a) # shape (batch, 1)
        # eval_net = Net()，所以也就是說，把現在的狀態做一批打包，丟去前向傳遞網路，然後回獲得所有動作的價值
        # q_eval 原本含有所有动作的值，然後针对做过的动作b_a, 来选 q_eval 的值
        # gather(1, b_a) --> 再根據所有動作的價值，選取我們當初施加那個動作的價值
        # 假設當初有向左、向右，如果選向左，那q_eval就是代表那時候向左的價值
        # eval_net輸入現在的狀態b_s後，就會生成我們所有動作的價值 --> # def forward(self, x)： return actions_value
        # dim = 1，也就是列固定，行變化，index索引就是行号。
        # gather在one - hot为输出的多分类问题中，可以把最大值坐标作为index传进去

        q_next = self.target_net(b_s_).detach() # q_next 不进行反向传递误差, 所以 detach
        # 將下一個狀態代入target_net，就可以得到q_next
        # 當我們學習的時候，所有的網路更新，可能會反向傳遞給我們的q_target，但是我們的q_target不希望它被更新，所以使用detach
        # q_target的更新是在這一步：
        # self.target_net.load_state_dict(self.eval_net.load_state_dict())
        # target_net的更新 --> target_net中的所有參數，從eval_net複製過來
        # eval_net在每一次‘def learn’的時候都在更新，而target_net是時不時的更新一下

        # (Q_target) - (當初所選取的動作價值)
        q_target = b_r + GAMMA * q_next.max(1)[0] # 最大值就是[0]，索引是[1]
        # shape (batch, 1)
        # q_target = 下一步q的價值 + 加上當初獲得的獎勵*對未來價值的遞減(GAMMA)
        # q_next包含從神經網路中分析出來的所有動作
        # q_next.max(1) --> 選擇當中最大的那一個動作
        # max返回的一個是最大值，一個是索引
        # torch.max(a,1) 返回每一列中最大值的那个元素，且返回索引（返回最大元素在这一行的列索引）
        # torch.max()[0]， 只返回最大值的每个数
        # troch.max()[1]， 只返回最大值的每个索引
        # torch.max(a,0) 返回每一行中最大值的那个元素，且返回索引（返回最大元素在这一列的行索引）

        # 利用預測值、真實值的誤差，進行神經網路的訓練 --> 歸零、反向傳遞、參數更新
        # 计算, 更新 eval net，
        loss = self.loss_func(q_eval, q_target) # 預測值q_eval(evaluation)，後面放真實值q_target
        self.optimizer.zero_grad() # 歸零
        loss.backward()            # 反向傳遞
        self.optimizer.step()      # 參數更新

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# 主程式開始
# 強化學習的過程，定义 DQN 系统
dqn = DQN()

print('\nCollecting experience...')
for i_episode in range(400):
    s = env.reset() # 設置成目前的環境狀態s
    # 跟環境互動，從環境得到的反饋；s代表state
    while True:
        env.render()                      # 環境渲染，显示实验动画
        a = dqn.choose_action(s)          # DQN根據現在的狀態s，來採取動作a; a代表action

        # take action
        s_, r, done, info = env.step(a)   # 根據採取的行為a去進行環境更新，得到環境給的反饋r、後來的狀態s_等
        # 如果更新完成，doen就等於True

        # 有些env, 可以想象成一个没有终止的环境, time_step只是时间上的结束, 但是在现实中, 可能并没有结束

        # 因為這個立桿子的遊戲，如果使用默認的reward返回的話，會有一點難學，所以要改掉預設的reward function
        # 設置桿子如果越偏向中間，reward就越小；車如果越偏向旁邊，reward就越小
        # 把桿子立起來是最大的reward；車在中間，reward最大
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians -abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        # 存记忆
        dqn.store_transition(s, a, r, s_) # DQN會存儲這些記憶來學習：之前的狀態、施加的動作、環境給的reward、環境導引去的下一個狀態

        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()                   # 记忆库满了就进行学习

        if done:                          # 如果回合结束, 进入下回合
            break                         # 如果更新完成，doen就等於True

        s = s_                            # 將現在的狀態，傳到下一回合的當做初始狀態


# 莫神，在您的代码中，episode终结的时候，并没有把q_target赋值为0。并不是标准的dqn，这样做为什么是合理的呢？
# 是否会导致两种结果，1.最终reword很低 2.最终reword看上去很高，但其实训练出来的模型完全是错的
# 经过试验，如果去掉您的reward warpper，就会出现 1.这样的情况
# 对, 我没有那么做, 我其实是假设 episode 只是时间上的结束, 而不是"死掉了/赢了"这种结束.
# 所以我把episode最后一步当做普通步处理了, 这样可以简化代码, 不过如果你觉得加上0的处理是必要的, 其实自己加上也无妨



