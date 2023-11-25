import torch
import torch.nn as nn
import torch.nn.functional as F
from util.net import *
from util.attention import MultiHeadAttention
# from algorithm.vdn_qmix import QMIX_Net
import numpy as np
import copy
from sklearn.cluster import KMeans
from torch.optim.lr_scheduler import StepLR
# from kmeans_pytorch import kmeans


class RECL_MIX(nn.Module):
    def __init__(self, args):
        super(RECL_MIX, self).__init__()
        self.args = args
        self.N = args.N
        self.state_dim = args.state_dim
        self.mix_input_dim = args.state_dim + args.N * args.att_out_dim
        self.batch_size = args.batch_size
        self.qmix_hidden_dim = args.qmix_hidden_dim
        self.hyper_hidden_dim = args.hyper_hidden_dim
        self.hyper_layers_num = args.hyper_layers_num

        self.state_fc = nn.Linear(args.state_dim, args.state_dim)
        self.state_gru = nn.GRUCell(args.state_dim, args.N*args.state_embed_dim)
        self.state_gru_hidden = None
        self.dim_q = args.state_embed_dim
        self.attention_net = MultiHeadAttention(args.n_heads, args.att_dim, args.att_out_dim, args.soft_temperature, self.dim_q,args.role_embedding_dim, args.role_embedding_dim)
        
                                       
        """
        w1:(N, qmix_hidden_dim)
        b1:(1, qmix_hidden_dim)
        w2:(qmix_hidden_dim, 1)
        b2:(1, 1)

        """
        if self.hyper_layers_num == 2:
            print("hyper_layers_num=2")
            self.hyper_w1 = nn.Sequential(nn.Linear(self.mix_input_dim, self.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hyper_hidden_dim, self.N * self.qmix_hidden_dim))
            self.hyper_w2 = nn.Sequential(nn.Linear(self.mix_input_dim, self.hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(self.hyper_hidden_dim, self.qmix_hidden_dim * 1))
        elif self.hyper_layers_num == 1:
            print("hyper_layers_num=1")
            self.hyper_w1 = nn.Linear(self.mix_input_dim, self.N * self.qmix_hidden_dim)
            self.hyper_w2 = nn.Linear(self.mix_input_dim, self.qmix_hidden_dim * 1)
        else:
            print("wrong!!!")

        self.hyper_b1 = nn.Linear(self.mix_input_dim, self.qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(self.mix_input_dim, self.qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.qmix_hidden_dim, 1))
    
    def role_gru_forward(self, role_embeddings):
        # role_embeddings.shape = (batch_size, N*role_embedding_dim)
        self.role_gru_hidden = self.role_gru(role_embeddings, self.role_gru_hidden)
        output = torch.sigmoid(self.role_gru_hidden)
        return output

    def forward(self, q, s, att):
        # q.shape(batch_size, max_episode_len, N)
        # s.shape(batch_size, max_episode_len,state_dim)
        
        q = q.view(-1, 1, self.N)  # (batch_size * max_episode_len, 1, N)
        s = s.reshape(-1, self.state_dim)  # (batch_size * max_episode_len, state_dim)
        att = att.reshape(-1, att.shape[2])
        state = torch.cat([s, att], dim=-1)

        w1 = torch.abs(self.hyper_w1(state))  # (batch_size * max_episode_len, N * qmix_hidden_dim)
        b1 = self.hyper_b1(state)  # (batch_size * max_episode_len, qmix_hidden_dim)
        w1 = w1.view(-1, self.N, self.qmix_hidden_dim)  # (batch_size * max_episode_len, N,  qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.qmix_hidden_dim)  # (batch_size * max_episode_len, 1, qmix_hidden_dim)

        # torch.bmm: 3 dimensional tensor multiplication
        q_hidden = F.elu(torch.bmm(q, w1) + b1)  # (batch_size * max_episode_len, 1, qmix_hidden_dim)

        w2 = torch.abs(self.hyper_w2(state))  # (batch_size * max_episode_len, qmix_hidden_dim * 1)
        b2 = self.hyper_b2(state)  # (batch_size * max_episode_len,1)
        w2 = w2.view(-1, self.qmix_hidden_dim, 1)  # (b\atch_size * max_episode_len, qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)  # (batch_size * max_episode_len, 1， 1)

        q_total = torch.bmm(q_hidden, w2) + b2  # (batch_size * max_episode_len, 1， 1)
        q_total = q_total.view(self.batch_size, -1, 1)  # (batch_size, max_episode_len, 1)
        return q_total

class RECL_NET(nn.Module):
    def __init__(self, args):
        super(RECL_NET, self).__init__()

        self.N = args.N
        self.agent_embedding_dim = args.agent_embedding_dim
        self.role_embedding_dim = args.role_embedding_dim
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim

        self.agent_embedding_net = Agent_Embedding(args)
        self.agent_embedding_decoder = Agent_Embedding_Decoder(args)
        self.role_embedding_net = Role_Embedding(args)
        self.role_embedding_target_net = Role_Embedding(args)
        self.role_embedding_target_net.load_state_dict(self.role_embedding_net.state_dict())

        self.W = nn.Parameter(torch.rand(self.role_embedding_dim, self.role_embedding_dim))

    def forward(self, obs, action, detach=False):
        agent_embedding = self.agent_embedding_net(obs, action, detach)
        role_embedding = self.role_embedding_net(agent_embedding)
        return role_embedding

    def encoder_decoder_forward(self, obs, action):
        agent_embedding = self.agent_embedding_forward(obs, action, detach=False)
        decoder_out = self.agent_embedding_decoder(agent_embedding)
        return decoder_out

    def agent_embedding_forward(self, obs, action, detach=False):
        return self.agent_embedding_net(obs, action, detach)
    
    def role_embedding_forward(self, agent_embedding, detach=False, ema=False):
        if ema:
            output = self.role_embedding_target_net(agent_embedding, detach)
        else:
            output = self.role_embedding_net(agent_embedding)
        return output
    
    def batch_role_embed_forward(self, batch_o, batch_a, max_episode_len, detach=False):
        self.agent_embedding_net.rnn_hidden = None
        agent_embeddings = []
        for t in range(max_episode_len+1):    # t = 0,1,2...(max_episode_len-1), max_episode_len
            agent_embedding = self.agent_embedding_forward(batch_o[:, t].reshape(-1, self.obs_dim),
                                          batch_a[:, t].reshape(-1, self.action_dim),
                                          detach=detach)  # agent_embedding.shape=(batch_size*N, agent_embed_dim)
            agent_embedding = agent_embedding.reshape(batch_o.shape[0], self.N, -1)  # shape=(batch_size,N, agent_embed_dim)
            agent_embeddings.append(agent_embedding.reshape(batch_o.shape[0],self.N, -1))
        # Stack them according to the time (dim=1)
        agent_embeddings = torch.stack(agent_embeddings, dim=1).reshape(-1,self.agent_embedding_dim) # agent_embeddings.shape=(batch_size*(max_episode_len+1)*N, agent_embed_dim)
        role_embeddings = self.role_embedding_forward(agent_embeddings, detach=False, ema=False).reshape(-1, max_episode_len+1, self.N, self.role_embedding_dim)
        agent_embeddings = agent_embeddings.reshape(-1, max_episode_len+1, self.N, self.agent_embedding_dim)
        return agent_embeddings, role_embeddings

class ACORM_Agent(object):
    def __init__(self, args):
        self.args = args
        self.N = args.N
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.role_embedding_dim = args.role_embedding_dim
        self.agent_embedding_dim = args.agent_embedding_dim
        self.att_out_dim = args.att_out_dim
        self.cluster_num = args.cluster_num
        self.add_last_action = args.add_last_action
        self.max_train_steps=args.max_train_steps
        self.lr = args.lr
        self.recl_lr = args.recl_lr
        self.agent_embedding_lr = args.agent_embedding_lr
        self.gamma = args.gamma
        
        self.batch_size = args.batch_size  
        self.multi_steps = args.multi_steps
        self.target_update_freq = args.target_update_freq
        self.train_recl_freq = args.train_recl_freq
        self.tau = args.tau
        self.role_tau = args.role_tau
        self.use_hard_update = args.use_hard_update
        self.use_lr_decay = args.use_lr_decay
        self.lr_decay_steps = args.lr_decay_steps
        self.lr_decay_rate = args.lr_decay_rate
        self.algorithm = args.algorithm        
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        self.QMIX_input_dim = args.obs_dim
        if self.add_last_action:
            print("------add last action------")
            self.QMIX_input_dim += self.action_dim
        self.QMIX_input_dim += self.role_embedding_dim

        self.RECL = RECL_NET(args)
        # self.agent_embedding_optimizer = torch.optim.Adam(self.RECL.agent_embedding_net.parameters(), lr=self.recl_lr)
        self.role_parameters = list(self.RECL.role_embedding_net.parameters()) + list(self.RECL.agent_embedding_net.parameters())
        self.role_embedding_optimizer = torch.optim.Adam(self.role_parameters, lr=self.lr)  
        self.role_lr_decay = StepLR(self.role_embedding_optimizer, step_size=self.lr_decay_steps, gamma=self.lr_decay_rate)
        self.RECL_parameters = list(self.RECL.parameters())
        self.RECL_optimizer = torch.optim.Adam(self.RECL_parameters, lr=self.recl_lr)
        self.encoder_decoder_para = list(self.RECL.agent_embedding_net.parameters()) + list(self.RECL.agent_embedding_decoder.parameters())
        self.encoder_decoder_optimizer = torch.optim.Adam(self.encoder_decoder_para, lr=self.agent_embedding_lr)
        
        self.eval_Q_net = Q_network_RNN(args, self.QMIX_input_dim)
        self.target_Q_net = Q_network_RNN(args, self.QMIX_input_dim)
        self.target_Q_net.load_state_dict(self.eval_Q_net.state_dict())

        self.eval_mix_net = RECL_MIX(args)
        self.target_mix_net = RECL_MIX(args)
        self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())
        
        self.eval_parameters = list(self.eval_mix_net.parameters()) + list(self.eval_Q_net.parameters())
        self.optimizer = torch.optim.Adam(self.eval_parameters, lr=self.lr)
        self.qmix_lr_decay = StepLR(self.optimizer, step_size=self.lr_decay_steps, gamma=self.lr_decay_rate)

        self.target_Q_net.to(self.device)
        self.eval_Q_net.to(self.device)
        self.target_mix_net.to(self.device)
        self.eval_mix_net.to(self.device)
        self.RECL.to(self.device)

        self.train_step = 0

    def get_role_embedding(self, obs_n, last_a):
        recl_obs = torch.tensor(np.array(obs_n), dtype=torch.float32).to(self.device)
        recl_last_a = torch.tensor(np.array(last_a), dtype=torch.float32).to(self.device)
        role_embedding = self.RECL(recl_obs, recl_last_a, detach=True)
        return role_embedding
    
    def choose_action(self, obs_n, last_onehot_a_n, role_embedding, avail_a_n, epsilon):
        with torch.no_grad():
            if np.random.uniform() < epsilon:  # epsilon-greedy
                # Only available actions can be chosen
                a_n = [np.random.choice(np.nonzero(avail_a)[0]) for avail_a in avail_a_n]
            else:
                inputs = copy.deepcopy(obs_n)
                if self.add_last_action:
                    inputs = np.hstack((inputs, last_onehot_a_n))
                inputs = np.hstack((inputs, role_embedding.to('cpu')))
                inputs = torch.tensor(inputs, dtype=torch.float32)
                inputs = inputs.to(self.device)

                q_value = self.eval_Q_net(inputs)
                avail_a_n = torch.tensor(avail_a_n, dtype=torch.float32)  # avail_a_n.shape=(N, action_dim)
                q_value = q_value.to('cpu')
                q_value[avail_a_n == 0] = -float('inf')  # Mask the unavailable actions

                a_n = q_value.argmax(dim=-1).numpy()
        return a_n
            
    def get_inputs(self, batch):
        inputs = copy.deepcopy(batch['obs_n'])
        if self.add_last_action:
            inputs = np.concatenate((inputs, batch['last_onehot_a_n']),axis=-1)
        inputs = torch.tensor(inputs, dtype=torch.float32)

        inputs = inputs.to(self.device)
        batch_o = batch['obs_n'].to(self.device)
        batch_s = batch['s'].to(self.device)
        batch_r = batch['r'].to(self.device)
        batch_a = batch['a_n'].to(self.device)
        batch_last_a = batch['last_onehot_a_n'].to(self.device)
        batch_active = batch['active'].to(self.device)
        batch_dw = batch['dw'].to(self.device)
        batch_avail_a_n = batch['avail_a_n']
        return inputs, batch_o, batch_s, batch_r, batch_a, batch_last_a, batch_avail_a_n, batch_active, batch_dw
    
    def train(self, replay_buffer):
        self.train_step += 1
        batch, max_episode_len = replay_buffer.sample(self.batch_size)  # Get training data
        inputs, batch_o, batch_s, batch_r, batch_a, batch_last_a, batch_avail_a_n, batch_active, batch_dw = self.get_inputs(batch)

        if self.train_step % self.train_recl_freq == 0:
            self.update_recl(batch_o, batch_last_a, batch_active, max_episode_len)
            self.soft_update_params(self.RECL.role_embedding_net, self.RECL.role_embedding_target_net, self.role_tau)

        self.update_qmix(inputs, batch_o, batch_s, batch_r, batch_a, batch_last_a, batch_avail_a_n, batch_active, batch_dw, max_episode_len)
        if self.use_hard_update:
            # hard update
            if self.train_step % self.target_update_freq == 0:
                self.target_Q_net.load_state_dict(self.eval_Q_net.state_dict())
                self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())
        else:
            # Softly update the target networks
            self.soft_update_params(self.eval_Q_net, self.target_Q_net, self.tau)
            self.soft_update_params(self.eval_mix_net, self.target_mix_net, self.tau)
        self.soft_update_params(self.RECL.role_embedding_net, self.RECL.role_embedding_target_net, self.tau)
    
        if self.use_lr_decay:
                self.qmix_lr_decay.step()
                self.role_lr_decay.step()

    def pretrain_recl(self, replay_buffer):
        batch, max_episode_len = replay_buffer.sample(self.batch_size)
        batch_o = batch['obs_n'].to(self.device)
        batch_last_a = batch['last_onehot_a_n'].to(self.device)
        batch_active = batch['active'].to(self.device)
        recl_loss = self.update_recl(batch_o, batch_last_a, batch_active, max_episode_len)
        self.soft_update_params(self.RECL.role_embedding_net, self.RECL.role_embedding_target_net, self.role_tau)
        return recl_loss
        

    def pretrain_agent_embedding(self, replay_buffer):
        batch, max_episode_len = replay_buffer.sample(self.batch_size)
        batch_o = batch['obs_n'].to(self.device)
        batch_last_a = batch['last_onehot_a_n'].to(self.device)
        batch_active = batch['active'].to(self.device)
        
        self.RECL.agent_embedding_net.rnn_hidden = None
        agent_embeddings = []
        for t in range(max_episode_len):
            agent_embedding = self.RECL.agent_embedding_forward(batch_o[:, t].reshape(-1, self.obs_dim),
                                            batch_last_a[:, t].reshape(-1, self.action_dim),
                                            detach=False) 
            agent_embeddings.append(agent_embedding.reshape(-1, self.N, self.agent_embedding_dim))   # (batch_size, N, agent_embedding_dim)
        agent_embeddings =  torch.stack(agent_embeddings, dim=1)    #(batch_size, max_episode_len, N, agent_embedding_dim)
        decoder_output = self.RECL.agent_embedding_decoder(agent_embeddings.reshape(-1,self.agent_embedding_dim)).reshape(-1, max_episode_len, self.N, self.obs_dim+self.N)
        batch_obs_hat = batch_o[:,1:]
        agent_id_one_hot = torch.eye(self.N).unsqueeze(0).unsqueeze(0).repeat(batch_o.shape[0], max_episode_len, 1, 1).to(self.device)
        decoder_target = torch.cat([batch_obs_hat, agent_id_one_hot], dim=-1)   # (batch_size, max_len, N, obs_dim+N)
        mask = batch_active.unsqueeze(-1).repeat(1, 1, self.N, self.obs_dim+self.N)
        loss = (((decoder_output - decoder_target) * mask)**2).sum()/mask.sum()
                
        self.encoder_decoder_optimizer.zero_grad()
        loss.backward()
        self.encoder_decoder_optimizer.step()
        return loss
        
    
    def update_recl(self, batch_o, batch_last_a, batch_active, max_episode_len):
        """
        N = agent_num
        batch_o.shape = (batch_size, max_episode_len + 1, N,  obs_dim)
        batch_a.shape = (batch_size, max_episode_len, N,  action_dim)
        batch_active = (batch_size, max_episode_len, 1)
        """
        self.RECL.agent_embedding_net.rnn_hidden = None
        loss = 0
        labels = np.zeros((batch_o.shape[0], self.N))  # (batch_size, N)
        for t in range(max_episode_len):    # t = 0,1,2...(max_episode_len-1)
            with torch.no_grad():
                agent_embedding = self.RECL.agent_embedding_forward(batch_o[:, t].reshape(-1, self.obs_dim),
                                            batch_last_a[:, t].reshape(-1, self.action_dim),
                                            detach=True)  # agent_embedding.shape=(batch_size*N, agent_embed_dim)
            role_embedding_qury = self.RECL.role_embedding_forward(agent_embedding,
                                                                   detach=False,
                                                                   ema=False).reshape(-1,self.N, self.role_embedding_dim)   # shape=(batch_size, N, role_embed_dim)
            role_embedding_key = self.RECL.role_embedding_forward(agent_embedding,
                                                                  detach=True,
                                                                  ema=True).reshape(-1,self.N, self.role_embedding_dim)
            logits = torch.bmm(role_embedding_qury, self.RECL.W.squeeze(0).expand((role_embedding_qury.shape[0],self.role_embedding_dim,self.role_embedding_dim)))
            logits = torch.bmm(logits, role_embedding_key.transpose(1,2))   # (batch_size, N, N)
            logits = logits - torch.max(logits, dim=-1)[0][:,:,None]
            exp_logits = torch.exp(logits) # (batch_size, N, 1)
            agent_embedding = agent_embedding.reshape(batch_o.shape[0],self.N, -1).to('cpu')  # shape=(batch_size,N, agent_embed_dim)

            for idx in range(agent_embedding.shape[0]): # idx = 0,1,2...(batch_size-1)
                if batch_active[idx, t] > 0.5:
                    if t % self.multi_steps == 0:
                        clusters_labels = KMeans(n_clusters=self.cluster_num).fit(agent_embedding[idx]).labels_ # (1,N)
                        labels[idx] = copy.deepcopy(clusters_labels)
                    else:
                        clusters_labels = copy.deepcopy(labels[idx])
                    # clusters_labels, _ = kmeans(X=agent_embedding[idx],num_clusters=self.cluster_num)
                    for j in range(self.cluster_num):   # j = 0,1,...(cluster_num -1)
                        label_pos = [idx for idx, value in enumerate(clusters_labels) if value==j]
                        # label_neg = [idx for idx, value in enumerate(clusters_labels) if value!=j]
                        for anchor in label_pos:
                            loss += -torch.log(exp_logits[idx, anchor, label_pos].sum()/exp_logits[idx, anchor].sum())
        loss /= (self.batch_size * max_episode_len * self.N)
        if batch_active[idx, t] > 0.5:
            self.RECL_optimizer.zero_grad()
            loss.backward()
            self.RECL_optimizer.step()
        return loss
    
    def update_qmix(self, inputs, batch_o, batch_s, batch_r, batch_a, batch_last_a, batch_avail_a_n, batch_active, batch_dw, max_episode_len):
        self.eval_Q_net.rnn_hidden = None
        self.target_Q_net.rnn_hidden = None
        _, role_embeddings = self.RECL.batch_role_embed_forward(batch_o, batch_last_a, max_episode_len, detach=False) # shape=(batch_size, (max_episode_len+1),N, role_embed_dim)
        inputs = torch.cat([inputs, role_embeddings], dim=-1)
        q_evals, q_targets = [], []

        self.eval_mix_net.state_gru_hidden = None
        # self.target_mix_net.state_gru_hidden = None
        fc_batch_s = F.relu(self.eval_mix_net.state_fc(batch_s.reshape(-1, self.state_dim))).reshape(-1, max_episode_len+1, self.state_dim)    # shape(batch*max_len+1, state_dim)
        state_gru_outs = []
        for t in range(max_episode_len):  # t=0,1,2,...(episode_len-1)
            q_eval = self.eval_Q_net(inputs[:, t].reshape(-1, self.QMIX_input_dim))  # q_eval.shape=(batch_size*N,action_dim)
            q_target = self.target_Q_net(inputs[:, t + 1].reshape(-1, self.QMIX_input_dim))
            q_evals.append(q_eval.reshape(self.batch_size, self.N, -1))  # q_eval.shape=(batch_size,N,action_dim)
            q_targets.append(q_target.reshape(self.batch_size, self.N, -1))

            self.eval_mix_net.state_gru_hidden = self.eval_mix_net.state_gru(fc_batch_s[:, t].reshape(-1,self.state_dim), self.eval_mix_net.state_gru_hidden)   # shape=(batch, N*state_embed_dim)
            state_gru_outs.append(self.eval_mix_net.state_gru_hidden)

        #     role_eval = self.eval_mix_net.role_gru_forward(role_embeddings[:,t].reshape(-1, self.N*self.role_embedding_dim))   # shape=(batch_size, N*role_embed_dim)
        #     role_target = self.target_mix_net.role_gru_forward(role_embeddings[:,t].reshape(-1, self.N*self.role_embedding_dim))   # shape=(batch_size, N*role_embed_dim)
        #     role_evals.append(role_eval)
        #     role_targets.append(role_target)
        self.eval_mix_net.state_gru_hidden = self.eval_mix_net.state_gru(fc_batch_s[:, max_episode_len].reshape(-1,self.state_dim), self.eval_mix_net.state_gru_hidden)
        state_gru_outs.append(self.eval_mix_net.state_gru_hidden)
        # role_targets.append(self.target_mix_net.role_gru_forward(role_embeddings[:,max_episode_len].reshape(-1, self.N*self.role_embedding_dim)))

        # Stack them according to the time (dim=1)
        # role_evals = torch.stack(role_evals, dim=1)     # shape=(batch_size, max_len+1, N*role_dim)
        # role_targets = torch.stack(role_targets, dim=1) 
        state_gru_outs = torch.stack(state_gru_outs, dim=1).reshape(-1, self.N, self.args.state_embed_dim) # shape=(batch*max_len+1, N,state_embed_dim)
        q_evals = torch.stack(q_evals, dim=1)  # q_evals.shape=(batch_size,max_episode_len,N,action_dim)
        q_targets = torch.stack(q_targets, dim=1)

        with torch.no_grad():
            q_eval_last = self.eval_Q_net(inputs[:, -1].reshape(-1, self.QMIX_input_dim)).reshape(self.batch_size, 1, self.N, -1)
            q_evals_next = torch.cat([q_evals[:, 1:], q_eval_last], dim=1) # q_evals_next.shape=(batch_size,max_episode_len,N,action_dim)
            q_evals_next[batch_avail_a_n[:, 1:] == 0] = -999999
            a_argmax = torch.argmax(q_evals_next, dim=-1, keepdim=True)  # a_max.shape=(batch_size,max_episode_len, N, 1)
            q_targets = torch.gather(q_targets, dim=-1, index=a_argmax).squeeze(-1)  # q_targets.shape=(batch_size, max_episode_len, N)
        q_evals = torch.gather(q_evals, dim=-1, index=batch_a.unsqueeze(-1)).squeeze(-1)  # q_evals.shape(batch_size, max_episode_len, N)
        
        role_embeddings = role_embeddings.reshape(-1, self.N, self.role_embedding_dim) # shape=((batch_size * max_episode_len+1), N, role_embed_dim)
        # eval_state_qv = self.eval_mix_net.state_fc2(batch_s.reshape(-1, self.state_dim)).reshape(-1, self.N, self.eval_mix_net.dim_k) # shape=(batch*max_len, N, state_dim//N)
        # target_state_qv = self.target_mix_net.state_fc2(batch_s.reshape(-1, self.state_dim)).reshape(-1, self.N, self.eval_mix_net.dim_k)
        # agent_embeddings = agent_embeddings.reshape(-1, self.N, self.agent_embedding_dim)
        att_eval = self.eval_mix_net.attention_net(state_gru_outs, role_embeddings, role_embeddings).reshape(-1, max_episode_len+1, self.N*self.att_out_dim) # ((batch*max_episode_len+1), N, att_dim)->(batch, len, N*att_dim)
        with torch.no_grad():
            att_target = self.target_mix_net.attention_net(state_gru_outs, role_embeddings, role_embeddings).reshape(-1, max_episode_len+1, self.N*self.att_out_dim) # ((batch*max_episode_len+1), N, att_dim)->(batch, len, N*att_dim)
        
        # eval_batch_s = self.eval_mix_net.state_fc1(batch_s.reshape(-1, self.state_dim)).reshape(-1, max_episode_len+1, self.state_dim)
        # traget_batch_s = self.target_mix_net.state_fc1(batch_s.reshape(-1, self.state_dim)).reshape(-1, max_episode_len+1, self.state_dim)

        q_total_eval = self.eval_mix_net(q_evals, fc_batch_s[:, :-1], att_eval[:, :-1])
        q_total_target = self.target_mix_net(q_targets, fc_batch_s[:, 1:], att_target[:, 1:])
        targets = batch_r + self.gamma * (1 - batch_dw) * q_total_target
        td_error = (q_total_eval - targets.detach())    # targets.detach() to cut the backward
        mask_td_error = td_error * batch_active
        loss = (mask_td_error ** 2).sum() / batch_active.sum()
        self.optimizer.zero_grad()
        self.role_embedding_optimizer.zero_grad()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.role_parameters, 10)
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, 10)        
        self.optimizer.step()
        self.role_embedding_optimizer.step()

    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
