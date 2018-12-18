import torch
from torch.autograd import Variable


class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, dropout = 0.2):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.rnn = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=dropout,
            bidirectional=False,
        )
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size)
        self.bn1 = torch.nn.BatchNorm1d(hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        batch_size = x.size()[1]
        seq_length = x.size()[0]
        #x = x.view(seq_length, batch_size, -1)

        h0 = Variable(torch.zeros(seq_length, batch_size, self.hidden_size))
        c0 = Variable(torch.zeros(seq_length, batch_size, self.hidden_size))
        outputs, (ht, ct) = self.rnn(x, (h0, c0))

        out = outputs[-1]
        out = self.bn1(self.fc1(out))
        out = self.relu(out)
        out = torch.nn.functional.dropout(out, training=self.training)
        out = self.fc2(out)

        return out