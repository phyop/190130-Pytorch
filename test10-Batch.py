import torch
import torch.utils.data as Data
torch.manual_seed(1) # reproducible

BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

# 先转换成 torch 能识别的 Dataset
torch_dataset = Data.TensorDataset(x,y) # x是要訓練的，y是要算誤差的
# torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)
# 版本關係，需要調整:torch_dataset = Data.TensorDataset(x,y)

loader = Data.DataLoader( # 使我們的訓練變成一小批一小批
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2 # 線程=2，提升效率
)

for epoch in range(3): # 總體訓練3次，而每一個epoch，會把這10筆資料，挑選其中5筆作為一次性訓練的方式來拆分
    for step, (batch_x, batch_y) in enumerate(loader): # enumerate：加上索引，第一個step就是1.，第二個step就是2.
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ', batch_x.numpy(), '| batch y: ', batch_y.numpy())


# Epoch:  0 | Step:  0 | batch x:  [ 5.  7. 10.  3.  4.] | batch y:  [6. 4. 1. 8. 7.]
# # --> 第一個epoc的第一批次計算，挑[ 5.  7. 10.  3.  4.]
# Epoch:  0 | Step:  1 | batch x:  [2. 1. 8. 9. 6.] | batch y:  [ 9. 10.  3.  2.  5.]
# Epoch:  1 | Step:  0 | batch x:  [ 4.  6.  7. 10.  8.] | batch y:  [7. 5. 4. 1. 3.]
# # --> 第二個epoc的第一批次計算，挑[ 4.  6.  7. 10.  8.]，所以代表shuffle=True的確有作用
# Epoch:  1 | Step:  1 | batch x:  [5. 3. 2. 1. 9.] | batch y:  [ 6.  8.  9. 10.  2.]
# Epoch:  2 | Step:  0 | batch x:  [ 4.  2.  5.  6. 10.] | batch y:  [7. 9. 6. 5. 1.]
# Epoch:  2 | Step:  1 | batch x:  [3. 9. 1. 8. 7.] | batch y:  [ 8.  2. 10.