import torch
import torch.nn.functional as F


train_num = 5000

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

optimizer = torch.optim.SGD(net.parameters(), lr=0.3)
loss_func = torch.nn.MSELoss()

file_train = open('dataset/train_data.csv','r')
line = file_train.readline()

for t in range(train_num):
    x = torch.FloatTensor(40).zero_()
    y = torch.FloatTensor(20).zero_()   

    for first in range(10):
        line = file_train.readline()
        if line == '':
            break
        line = line.split(',')

        x[4*first] = float(line[6])
        x[4*first+1] = float(line[7])/10000
        x[4*first+2] = float(line[8])
        x[4*first+3] = float(line[9])/10000
    
    for second in range(20):
        line = file_train.readline()
        if line == '':
            break
        line = line.split(',')

        y[second] = float(line[3])

    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()     

    #print('predict: ',prediction.data.numpy(),'\nactual: ',y.data.numpy(),'\n')

file_train.close()
torch.save(net.state_dict(),'net_data_multi.pkl')
print('train complete: ',train_num)