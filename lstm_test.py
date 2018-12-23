import torch
import numpy as np
import math
import LSTM
import random


input_size = 1
hidden_size = 16

net = LSTM.LSTM(input_size, hidden_size)
net.load_state_dict(torch.load('data_lstm.pkl'))

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
    series = []
    series_sum = 0
    x = torch.FloatTensor(10).zero_()

    for i in range(10):
        line = file_test.readline().split(',')
        series.append(float(line[3]))
        series_sum += float(line[3])
    ave = series_sum/10
    end1 = series[8]
    end2 = series[9]

    for i in range(10):
        x[i] = series[i] - ave
    x = x[np.newaxis, :, np.newaxis]
    
    prediction = net(x).data.numpy()[-1][0] + end2 + random.uniform(-0.0001,0.0001)
    #print(end1, end2, prediction)

    file_out.write(str(case)+','+str(prediction)+'\n')

    line = file_test.readline()
    case += 1

file_test.close()
file_out.close()
