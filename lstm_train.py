import torch
import numpy as np
import LSTM


input_size = 1
hidden_size = 16

train_num = 10000


net = LSTM.LSTM(input_size, hidden_size)
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()


file_train = open('dataset/train_data.csv','r')
line = file_train.readline()

for t in range(train_num):
    x = torch.FloatTensor(10).zero_()
    for i in range(10):
        line = file_train.readline().split(',')
        x[i] = float(line[3])
    x = x[np.newaxis, :, np.newaxis]

    y = torch.FloatTensor(10).zero_()
    for j in range(10):
        line = file_train.readline().split(',')
        y[j] = float(line[3])
    y = y[:, np.newaxis]

    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()     

    '''
    predict = 0
    for i in range(10):
        predict += prediction[i][0]
    predict = predict/10
    print('predict: ',predict)
    '''

file_train.close()
torch.save(net.state_dict(),'net_data_lstm.pkl')
print('train complete: ',train_num)