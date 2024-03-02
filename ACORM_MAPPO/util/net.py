import torch
import torch.nn as nn

def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)

class Actor(nn.Module):
    def __init__(self, args, input_dim):
        super(Actor, self).__init__()
        self. rnn_hidden = None
        self.fc1 = nn.Linear(input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.action_dim)
        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2, gain=0.01)

    def forward(self, input, avail_a_n):
        # When 'choose_action': input.shape=(N, input_dim), prob.shape=(N, action_dim)
        # When 'train':         input.shape=(batch*N, input_dim),prob.shape=(batch*N, action_dim)
        x = torch.relu(self.fc1(input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        x = self.fc2(self.rnn_hidden)
        x[avail_a_n==0] = -1e10         # mask the unavailable actions
        prob = torch.softmax(x, dim=-1)
        return prob
    

class Critic(nn.Module):
    def __init__(self, args, input_dim):
        super(Critic, self).__init__()
        self.rnn_hidden = None
        self.fc1 = nn.Linear(input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)
        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, input):
        # When 'get_value': input.shape=(N, input_dim), value.shape=(N, 1)
        # When 'train':     input.shape=(batch*N, input_dim), value.shape=(batch_size*N, 1)
        x = torch.relu(self.fc1(input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        value = self.fc2(self.rnn_hidden)
        return value
