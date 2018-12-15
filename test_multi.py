import torch
import torch.nn.functional as F
import csv


class Net(torch.nn.Module):

    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        h1 = F.relu(self.hidden(x))
        y = self.predict(h1)      
        return y


net = Net(n_feature=40, n_hidden=10, n_output=20)

net.load_state_dict(torch.load('net_data_multi.pkl'))

file_test = open('dataset/test_data.csv','r')
line = file_test.readline()

file_out = open('result_multi.csv','w')
file_out.write('caseid,midprice\n')

case = 1

while case < 143:
    line = file_test.readline().split(',')
    if len(line) < 9:
        case += 1
        
while case <= 153:
    x = torch.FloatTensor(40).zero_()
    y = torch.FloatTensor(20).zero_()

    for ct in range(10):
        line = file_test.readline()
        if line == '':
            break
        line = line.split(',')

        x[ct*4] = float(line[6])
        x[ct*4+1] = float(line[7])/10000
        x[ct*4+2] = float(line[8])
        x[ct*4+3] = float(line[9])/10000

    prediction = net(x)

    average = 0
    for k in range(10):
        average += prediction.data.numpy()[k]
    average = 1.0*average/10

    file_out.write(str(case)+','+str(average)+'\n')
    #print(str(case)+','+str(average)+'\n')

    line = file_test.readline()
    case += 1

file_test.close()
file_out.close()
print('test complete')
