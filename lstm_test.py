import torch
import numpy as np
import LSTM

input_size = 1
hidden_size = 16

net = LSTM.LSTM(input_size, hidden_size)
net.load_state_dict(torch.load('net_data_lstm.pkl'))

file_test = open('dataset/test_data.csv','r')
line = file_test.readline()

file_out = open('result_lstm.csv','w')
file_out.write('caseid,midprice\n')

case = 1
while case < 143:
    line = file_test.readline().split(',')
    if len(line) < 9:
        case += 1
        
while case <= 1000:
    x = torch.FloatTensor(10).zero_()
    for i in range(10):
        line = file_test.readline().split(',')
        x[i] = float(line[3])
    x = x[np.newaxis, :, np.newaxis]

    prediction = net(x)

    predict = 0
    for i in range(10):
        predict += prediction[i][0]
    predict = predict/10
    predict = predict.data.numpy()

    #print('predict: ',prediction[-1][0])
    file_out.write(str(case)+','+str(predict)+'\n')

    line = file_test.readline()
    case += 1

file_test.close()
file_out.close()
print('test complete')
