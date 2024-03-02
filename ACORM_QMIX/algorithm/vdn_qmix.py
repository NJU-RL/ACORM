import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util.net import Q_network_MLP, Q_network_RNN
import copy

class QMIX_Net(nn.Module):
    def __init__(self, args):
        super(QMIX_Net, self).__init__()
        self.N = args.N
        self.state_dim = args.state_dim
        self.batch_size = args.batch_size
        self.qmix_hidden_dim = args.qmix_hidden_dim
        self.hyper_hidden_dim = args.hyper_hidden_dim
        self.hyper_layers_num = args.hyper_layers_num
        """
        w1:(N, qmix_hidden_dim)
        b1:(1, qmix_hidden_dim)
        w2:(qmix_hidden_dim, 1)
        b2:(1, 1)

        """
        if self.hyper_layers_num == 2:
            print("hyper_layers_num=2")
            self.hyper_w1 = nn.Sequential(nn.Linear(self.state_dim, self.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hyper_hidden_dim, self.N * self.qmix_hidden_dim))
            self.hyper_w2 = nn.Sequential(nn.Linear(self.state_dim, self.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hyper_hidden_dim, self.qmix_hidden_dim * 1))
        elif self.hyper_layers_num == 1:
            print("hyper_layers_num=1")
            self.hyper_w1 = nn.Linear(self.state_dim, self.N * self.qmix_hidden_dim)
            self.hyper_w2 = nn.Linear(self.state_dim, self.qmix_hidden_dim * 1)
        else:
            print("wrong!!!")

        self.hyper_b1 = nn.Linear(self.state_dim, self.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(self.state_dim, self.qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.qmix_hidden_dim, 1))

    def forward(self, q, s):
        # q.shape(batch_size, max_episode_len, N)
        # s.shape(batch_size, max_episode_len,state_dim)
        q = q.view(-1, 1, self.N)  # (batch_size * max_episode_len, 1, N)
        s = s.reshape(-1, self.state_dim)  # (batch_size * max_episode_len, state_dim)

        w1 = torch.abs(self.hyper_w1(s))  # (batch_size * max_episode_len, N * qmix_hidden_dim)
        b1 = self.hyper_b1(s)  # (batch_size * max_episode_len, qmix_hidden_dim)
        w1 = w1.view(-1, self.N, self.qmix_hidden_dim)  # (batch_size * max_episode_len, N,  qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.qmix_hidden_dim)  # (batch_size * max_episode_len, 1, qmix_hidden_dim)

        # torch.bmm: 3 dimensional tensor multiplication
        q_hidden = F.elu(torch.bmm(q, w1) + b1)  # (batch_size * max_episode_len, 1, qmix_hidden_dim)

        w2 = torch.abs(self.hyper_w2(s))  # (batch_size * max_episode_len, qmix_hidden_dim * 1)
        b2 = self.hyper_b2(s)  # (batch_size * max_episode_len,1)
        w2 = w2.view(-1, self.qmix_hidden_dim, 1)  # (batch_size * max_episode_len, qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)  # (batch_size * max_episode_len, 1， 1)

        q_total = torch.bmm(q_hidden, w2) + b2  # (batch_size * max_episode_len, 1， 1)
        q_total = q_total.view(self.batch_size, -1, 1)  # (batch_size, max_episode_len, 1)
        return q_total


class VDN_Net(nn.Module):
    def __init__(self, ):
        super(VDN_Net, self).__init__()

    def forward(self, q):
        return torch.sum(q, dim=-1, keepdim=True)  # (batch_size, max_episode_len, 1)


class VDN_QMIX(object):
    def __init__(self, args):
        self.N = args.N
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.add_last_action = args.add_last_action
        self.add_agent_id = args.add_agent_id
        self.max_train_steps=args.max_train_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.use_grad_clip = args.use_grad_clip
        self.batch_size = args.batch_size  
        self.target_update_freq = args.target_update_freq
        self.tau = args.tau
        self.use_hard_update = args.use_hard_update
        self.use_rnn = args.use_rnn
        self.algorithm = args.algorithm
        self.use_double_q = args.use_double_q
        self.use_RMS = args.use_RMS
        self.use_lr_decay = args.use_lr_decay
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        self.use_gpu = args.use_gpu
        # Compute the input dimension
        self.input_dim = self.obs_dim
        if self.add_last_action:
            print("------add last action------")
            self.input_dim += self.action_dim
        if self.add_agent_id:
            print("------add agent id------")
            self.input_dim += self.N

        if self.use_rnn:
            print("------use RNN------")
            self.eval_Q_net = Q_network_RNN(args, self.input_dim)
            self.target_Q_net = Q_network_RNN(args, self.input_dim)
        else:
            print("------use MLP------")
            self.eval_Q_net = Q_network_MLP(args, self.input_dim)
            self.target_Q_net = Q_network_MLP(args, self.input_dim)
        self.target_Q_net.load_state_dict(self.eval_Q_net.state_dict())

        if self.algorithm == "QMIX":
            print("------algorithm: QMIX------")
            self.eval_mix_net = QMIX_Net(args)
            self.target_mix_net = QMIX_Net(args)
        elif self.algorithm == "VDN":
            print("------algorithm: VDN------")
            self.eval_mix_net = VDN_Net()
            self.target_mix_net = VDN_Net()
        else:
            print("wrong!!!")
        
        self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())
        self.eval_parameters = list(self.eval_mix_net.parameters()) + list(self.eval_Q_net.parameters())
        
        if self.use_RMS:
            print("------optimizer: RMSprop------")
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=self.lr)
        else:
            print("------optimizer: Adam------")
            self.optimizer = torch.optim.Adam(self.eval_parameters, lr=self.lr)
        
        if self.use_gpu:
            self.target_Q_net.to(self.device)
            self.eval_Q_net.to(self.device)
            self.target_mix_net.to(self.device)
            self.eval_mix_net.to(self.device)
        self.train_step = 0

    def choose_action(self, obs_n, last_onehot_a_n, avail_a_n, epsilon):
        with torch.no_grad():
            if np.random.uniform() < epsilon:  # epsilon-greedy
                # Only available actions can be chosen
                a_n = [np.random.choice(np.nonzero(avail_a)[0]) for avail_a in avail_a_n]
            else:
                # inputs = []
                # obs_n = torch.tensor(obs_n, dtype=torch.float32)  # obs_n.shape=(N，obs_dim)
                # inputs.append(obs_n)
                inputs = copy.deepcopy(obs_n)
                if self.add_last_action:
                    # last_a_n = torch.tensor(last_onehot_a_n, dtype=torch.float32)
                    # inputs.append(last_a_n)
                    inputs = np.hstack((inputs, last_onehot_a_n))
                if self.add_agent_id:
                    inputs = np.hstack((inputs, np.eye(self.N)))
                    # inputs.append(torch.eye(self.N))

                # inputs = torch.cat([x for x in inputs], dim=-1)  # inputs.shape=(N,inputs_dim)
                inputs = torch.tensor(inputs, dtype=torch.float32)  # nputs.shape = (N, obs_dim+action_dim+N)
                if self.use_gpu:
                    inputs = inputs.to(self.device)
                q_value = self.eval_Q_net(inputs)   

                avail_a_n = torch.tensor(avail_a_n, dtype=torch.float32)  # avail_a_n.shape=(N, action_dim)
                if self.use_gpu:
                    q_value = q_value.to('cpu')
                q_value[avail_a_n == 0] = -float('inf')  # Mask the unavailable actions
                a_n = q_value.argmax(dim=-1).numpy()
            return a_n

    def train(self, replay_buffer):
        batch, max_episode_len = replay_buffer.sample()  # Get training data
        self.train_step += 1

        inputs = self.get_inputs(batch, max_episode_len)  # inputs.shape=(bach_size,max_episode_len+1,N,input_dim)
        if self.use_gpu:
            inputs = inputs.to(self.device)
            batch_s = batch['s'].to(self.device)
            batch_r = batch['r'].to(self.device)
            batch_n = batch['a_n'].to(self.device)
            batch_active = batch['active'].to(self.device)
            batch_dw = batch['dw'].to(self.device)
        if self.use_rnn:
            self.eval_Q_net.rnn_hidden = None
            self.target_Q_net.rnn_hidden = None
            q_evals, q_targets = [], []
            for t in range(max_episode_len):  # t=0,1,2,...(episode_len-1)
                q_eval = self.eval_Q_net(inputs[:, t].reshape(-1, self.input_dim))  # q_eval.shape=(batch_size*N,action_dim)
                q_target = self.target_Q_net(inputs[:, t + 1].reshape(-1, self.input_dim))
                q_evals.append(q_eval.reshape(self.batch_size, self.N, -1))  # q_eval.shape=(batch_size,N,action_dim)
                q_targets.append(q_target.reshape(self.batch_size, self.N, -1))

            # Stack them according to the time (dim=1)
            q_evals = torch.stack(q_evals, dim=1)  # q_evals.shape=(batch_size,max_episode_len,N,action_dim)
            q_targets = torch.stack(q_targets, dim=1)
        else:
            q_evals = self.eval_Q_net(inputs[:, :-1])  # q_evals.shape=(batch_size,max_episode_len,N,action_dim)
            q_targets = self.target_Q_net(inputs[:, 1:])    # inputs[:, 1:] -> obs_next

        with torch.no_grad():
            if self.use_double_q:  # If use double q-learning, we use eval_net to choose actions,and use target_net to compute q_target
                q_eval_last = self.eval_Q_net(inputs[:, -1].reshape(-1, self.input_dim)).reshape(self.batch_size, 1, self.N, -1)
                q_evals_next = torch.cat([q_evals[:, 1:], q_eval_last], dim=1) # q_evals_next.shape=(batch_size,max_episode_len,N,action_dim)
                q_evals_next[batch['avail_a_n'][:, 1:] == 0] = -999999
                a_argmax = torch.argmax(q_evals_next, dim=-1, keepdim=True)  # a_max.shape=(batch_size,max_episode_len, N, 1)
                q_targets = torch.gather(q_targets, dim=-1, index=a_argmax).squeeze(-1)  # q_targets.shape=(batch_size, max_episode_len, N)
            else:
                q_targets[batch['avail_a_n'][:, 1:] == 0] = -999999     # batch['avail_a_n'].shape = (batch_size, max_episode_len, N, action_dim)
                q_targets = q_targets.max(dim=-1)[0]  # q_targets.shape=(batch_size, max_episode_len, N)

        # batch['a_n'].shape(batch_size,max_episode_len, N)
        q_evals = torch.gather(q_evals, dim=-1, index=batch_n.unsqueeze(-1)).squeeze(-1)  # q_evals.shape(batch_size, max_episode_len, N)

        # Compute q_total using QMIX or VDN, q_total.shape=(batch_size, max_episode_len, 1)
        if self.algorithm == "QMIX":
            q_total_eval = self.eval_mix_net(q_evals, batch_s[:, :-1])
            q_total_target = self.target_mix_net(q_targets, batch_s[:, 1:])
        else:
            q_total_eval = self.eval_mix_net(q_evals)
            q_total_target = self.target_mix_net(q_targets)
        # targets.shape=(batch_size,max_episode_len,1)
        
        targets = batch_r + self.gamma * (1 - batch_dw) * q_total_target

        td_error = (q_total_eval - targets.detach())    # targets.detach() to cut the backward
        mask_td_error = td_error * batch_active
        loss = (mask_td_error ** 2).sum() / batch_active.sum()
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.eval_parameters, 10)
        self.optimizer.step()

        if self.use_hard_update:
            # hard update
            if self.train_step % self.target_update_freq == 0:
                self.target_Q_net.load_state_dict(self.eval_Q_net.state_dict())
                self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())
        else:
            # Softly update the target networks
            for param, target_param in zip(self.eval_Q_net.parameters(), self.target_Q_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.eval_mix_net.parameters(), self.target_mix_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def get_inputs(self, batch, max_episode_len):
        # inputs = []
        # inputs.append(batch['obs_n'])
        inputs = copy.deepcopy(batch['obs_n'])  # batch['obs_n'].shape = (batch_size,max_episode_len+1, N, obs_dim)
        if self.add_last_action:
            inputs = np.concatenate((inputs, batch['last_onehot_a_n']),axis=-1)
            # inputs.append(batch['last_onehot_a_n'])
        if self.add_agent_id:
            agent_id_one_hot = torch.eye(self.N).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, max_episode_len + 1, 1, 1)
            # inputs.append(agent_id_one_hot)
            inputs = np.concatenate((inputs, agent_id_one_hot),axis=-1)
        inputs = torch.tensor(inputs, dtype=torch.float32)
        # inputs.shape=(bach_size,max_episode_len+1,N,input_dim)
        # inputs = torch.cat([x for x in inputs], dim=-1)
        return inputs

    def save_model(self, env_name, algorithm, seed, total_steps):
        torch.save(self.eval_Q_net.state_dict(), "./model/{}/{}_seed_{}_step_{}k.pth".format(env_name, algorithm, seed, int(total_steps / 1000)))
