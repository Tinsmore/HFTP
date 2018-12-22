import torch
import numpy as np
import math
import LSTM


input_size = 1
hidden_size = 16

train_num = 13000

net = LSTM.LSTM(input_size, hidden_size)
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.8)
loss_func = torch.nn.MSELoss()

file_train = open('dataset/train_data.csv','r')
line = file_train.readline()

for t in range(train_num):
    series = []
    series_sum = 0
    x = torch.FloatTensor(10).zero_()
    y = torch.FloatTensor(10).zero_()

    for i in range(10):
        line = file_train.readline().split(',')
        series.append(float(line[3]))
        series_sum += float(line[3])
    ave = series_sum/10

    for i in range(10):
        series_sum += (series[i] - ave)*(series[i] - ave)
    mse = series_sum/10

    for i in range(10):
        series[i] = (series[i] - ave)/math.sqrt(mse) #
        x[i] = series[i]

    for  i in range(9):
        y[i] = x[i+1]
    line = file_train.readline().split(',')
    y[9] = (float(line[3]) - ave)/math.sqrt(mse)

    x = x[np.newaxis, :, np.newaxis]
    y = y[:, np.newaxis]

    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()    

    #print('y: ',y.data.numpy()[0],' pre: ',prediction.data.numpy()[0], ' loss: ',loss.data.numpy())

file_train.close()
torch.save(net.state_dict(),'data_lstm.pkl')
print('train complete: ',train_num)