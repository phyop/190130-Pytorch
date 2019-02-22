# TensorFlow會自動調用最佳資源
# PyTorch要自己指定要用GPU還是CPU來運算

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.torchvision
from torch.autograd import Variable
import torchvision

# torch.manual_seed(1)

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

# train_data是裝載在train_loader上，方便進行批訓練使用
# 所以跑訓練的for迴圈epoch，就是去跑train_loader
# (x, y) in enumerate(train_loader) --> 就是mnist上的(手寫圖片，對應的真實數字)
# 為了要讓train_loader去GPU跑訓練，Variable(x)、Variable(y)就要移動到cuda()去
# 如同train_data一樣，我們也可以把test_data也轉移到GPU去，也就是test_x後面也加上cuda()

# 此外，pred_y = torch.max(test_output, 1)[1].cuda().data
# 裡面的test_output雖然是GPU的產品，可是當做引數丟給了torch.max之後，就變成可能是CPU的計算型式
# 所以GPU產品丟到torch計算後，都還是要加上.cuda()，確定又回轉到GPU
# 所以經過torch.max以及指定轉移到cuda()後，出來的pred_y就變成是GPU裡面的y了
# pred_y如果想要轉成CPU --> pred_y = pred_y.cpu()

# GPU型式是不能用matplot的，必須轉成CPU，而且要是numpy型式

# cnn = CNN()這個物件，也要移動到GPU上面 --> 所以後面要多加上這行 cnn.cuda()


train_data = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST,)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

# !!!!!!!! Change in here !!!!!!!!! #
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000].cuda()/255.   # Tensor on GPU
test_y = test_data.test_labels[:2000].cuda()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2,),
                                   nn.ReLU(), nn.MaxPool2d(kernel_size=2),)
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2),)
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


cnn = CNN()

# !!!!!!!! Change in here !!!!!!!!! #
cnn.cuda()      # Moves all model parameters and buffers to the GPU.

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):

        # !!!!!!!! Change in here !!!!!!!!! #
        b_x = Variable(x).cuda()    # Tensor on GPU
        b_y = Variable(y).cuda()    # Tensor on GPU

        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = cnn(test_x)

            # !!!!!!!! Change in here !!!!!!!!! #
            pred_y = torch.max(test_output, 1)[1].cuda().data  # move the computation in GPU

            accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)


test_output = cnn(test_x[:10])

# !!!!!!!! Change in here !!!!!!!!! #
pred_y = torch.max(test_output, 1)[1].cuda().data # move the computation in GPU

print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
