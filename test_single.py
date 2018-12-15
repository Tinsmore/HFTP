import torch
import torch.nn.functional as F
import csv


class Net(torch.nn.Module):

    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        inter = F.relu(self.hidden(x))
        y = self.predict(inter)      
        return y


net = Net(n_feature=4, n_hidden=10, n_output=1)

net.load_state_dict(torch.load('net_data_single.pkl'))

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()

file_test = open('dataset/test_data.csv','r')
line = file_test.readline()

file_out = open('result_single.csv','w')
file_out.write('caseid,midprice\n')

x = torch.FloatTensor(4).zero_()
y = torch.FloatTensor(1).zero_()

case = 1
last_predict = 0

while case < 143:
    line = file_test.readline().split(',')
    if len(line) < 9:
        case += 1
        
while True:
    line = file_test.readline()
    if line == '':
        break
    line = line.split(',')

    if len(line) > 9:
        x[0] = float(line[6])
        x[1] = float(line[7])/10000
        x[2] = float(line[8])
        x[3] = float(line[9])/10000
        y[0] = float(line[3])
    else:
        file_out.write(str(case)+','+str(last_predict)+'\n')
        case += 1
        continue

    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()     

    last_predict = prediction.data.numpy()[0]
    #print(y.data.numpy(),prediction.data.numpy())

file_test.close()
file_out.close()
