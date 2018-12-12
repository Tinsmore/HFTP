import torch
import torch.nn.functional as F


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

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()

file_train = open('dataset/train_data.csv','r')
line = file_train.readline()

x = torch.FloatTensor(4).zero_()
y = torch.FloatTensor(1).zero_()

for t in range(500):
    line = file_train.readline().split(',')

    x[0] = float(line[6])
    x[1] = float(line[7])/10000
    x[2] = float(line[8])
    x[3] = float(line[9])/10000
    y[0] = float(line[3])

    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()     

file_train.close()
torch.save(net.state_dict(),'net_data.pkl')
print('train complete')