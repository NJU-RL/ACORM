import torch
from torch.optim.lr_scheduler import StepLR
from torch.distributions import Categorical
from torch.utils.data.sampler import *
import numpy as np
import copy
from util.acorm_net import ACORM_Actor, ACORM_Critic
from sklearn.cluster import KMeans


class ACORM(object):
    def __init__(self, args):
        self.N = args.N
        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.state_dim = args.state_dim
        self.agent_embedding_dim = args.agent_embedding_dim
        self.role_embedding_dim = args.role_embedding_dim
        self.rnn_hidden_dim = args.rnn_hidden_dim

        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps

        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.lr = args.lr
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.clip_epsilon = args.clip_epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.tau = args.tau

        self.use_lr_decay = args.use_lr_decay
        self.lr_decay_steps = args.lr_decay_steps
        self.lr_decay_rate = args.lr_decay_rate
        self.use_adv_norm = args.use_adv_norm
        self.use_grad_clip = args.use_grad_clip

        # recl
        self.cl_lr = args.cl_lr
        self.multi_steps = args.multi_steps
        self.train_recl_freq = args.train_recl_freq
        self.cluster_num = args.cluster_num

        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        self.actor = ACORM_Actor(args)
        self.critic =ACORM_Critic(args)

        # self.actor_parameters = list(self.actor.actor_net.parameters()) \
        #                       + list(self.actor.embedding_net.agent_embedding_net.parameters()) \
        #                       + list(self.actor.embedding_net.role_embedding_net.encoder.parameters())                
        # self.actor_optimizer = torch.optim.Adam(self.actor_parameters, lr=self.actor_lr)
        # self.actor_lr_decay = StepLR(self.actor_optimizer, step_size=self.lr_decay_steps, gamma=self.lr_decay_rate)

        # self.critic_parameters = self.critic.parameters()
        # self.critic_optimizer = torch.optim.Adam(self.critic_parameters, lr=self.critic_lr)
        # self.critic_lr_decay = StepLR(self.critic_optimizer, step_size=self.lr_decay_steps, gamma=self.lr_decay_rate)

        self.encoder_decoder_para = list(self.actor.embedding_net.agent_embedding_net.parameters()) \
                                       + list(self.actor.embedding_net.agent_embedding_decoder.parameters())
        self.encoder_decoder_optimizer = torch.optim.Adam(self.encoder_decoder_para, lr=args.agent_embedding_lr)

        self.ac_parameters = list(self.actor.actor_net.parameters()) \
                              + list(self.actor.embedding_net.agent_embedding_net.parameters()) \
                              + list(self.actor.embedding_net.role_embedding_net.encoder.parameters()) \
                              + list(self.critic.parameters())           
        self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr)
        self.ac_lr_decay = StepLR(self.ac_optimizer, step_size=self.lr_decay_steps, gamma=self.lr_decay_rate)

        self.cl_parameters = self.actor.embedding_net.parameters()
        self.cl_optimizer = torch.optim.Adam(self.cl_parameters, lr=self.cl_lr)
        self.cl_lr_decay = StepLR(self.cl_optimizer, step_size=self.lr_decay_steps, gamma=self.lr_decay_rate)

        self.actor.to(self.device)
        self.critic.to(self.device)

        self.train_step = 0

    def choose_action(self, agent_embedding, role_embedding, avail_a_n, evaluate):
        with torch.no_grad():
            avail_a_n = torch.tensor(avail_a_n, dtype=torch.float32)  # avail_a_n.shape=(N, action_dim)
            avail_a_n = avail_a_n.to(self.device)
            prob = self.actor.actor_forward(agent_embedding, role_embedding, avail_a_n)

            if evaluate:
                a_n = prob.argmax(dim=-1).to('cpu')
                return a_n.numpy(), None
            else:
                dist = Categorical(probs=prob)
                a_n = dist.sample()
                a_logprob_n = dist.log_prob(a_n)
                return a_n.to('cpu').numpy(), a_logprob_n.to('cpu').numpy()
  
    def get_value(self, s, obs_n, role_embed_n):
        with torch.no_grad():
            # obs_n = torch.tensor(obs_n,dtype=torch.float32).to(self.device)   # (N, obs_dim)
            state = torch.tensor(np.array(s), dtype=torch.float32).unsqueeze(0).to(self.device)   # (state_dim,)->(1, state_dim)
            v_n = self.critic(obs_n, state, role_embed_n)
            return v_n.to('cpu').numpy().flatten()
        
    def train(self, replay_buffer):
        self.train_step += 1
        batch = replay_buffer.get_training_data()
        max_episode_len = replay_buffer.max_episode_len
        batch_obs, batch_s, batch_r, batch_v_n, batch_dw, batch_active, batch_avail_a_n, batch_a_n, batch_a_logprob_n = self.get_inputs(batch)

        if self.train_step % self.train_recl_freq == 0:
            self.update_recl(batch_obs, batch_active, max_episode_len)
            self.soft_update_params(self.actor.embedding_net.role_embedding_net.encoder, self.actor.embedding_net.role_embedding_net.target_encoder, self.tau)
        actor_loss, critic_loss = self.update_ppo(max_episode_len, batch_obs, batch_s, batch_r, batch_v_n, batch_dw, batch_active, batch_avail_a_n, batch_a_n, batch_a_logprob_n)
        self.soft_update_params(self.actor.embedding_net.role_embedding_net.encoder, self.actor.embedding_net.role_embedding_net.target_encoder, self.tau)
        return actor_loss, critic_loss

    def pretrain_agent_embedding(self, replay_buffer):
        batch = replay_buffer.get_training_data()
        max_episode_len = replay_buffer.max_episode_len
        batch_o = batch['obs_n'].to(self.device)        # (batch, max_len, N, obs_dim)
        batch_active = batch['active'].to(self.device)  # (batch, max_len, N)
        
        self.actor.embedding_net.agent_embedding_net.rnn_hidden = None
        agent_embeddings = []
        for t in range(max_episode_len-1):
            agent_embedding = self.actor.embedding_net.agent_embed_forward(batch_o[:, t].reshape(-1, self.obs_dim), 
                                                                           detach=False)
            agent_embeddings.append(agent_embedding.reshape(-1, self.N, self.agent_embedding_dim))   # (batch_size, N, agent_embedding_dim)
        agent_embeddings =  torch.stack(agent_embeddings, dim=1)    #(batch_size, max_episode_len, N, agent_embedding_dim)
        decoder_output = self.actor.embedding_net.agent_embedding_decoder(agent_embeddings.reshape(-1,self.agent_embedding_dim)).reshape(-1, max_episode_len-1, self.N, self.obs_dim+self.N)
        batch_obs_hat = batch_o[:,1:]
        agent_id_one_hot = torch.eye(self.N).unsqueeze(0).unsqueeze(0).repeat(batch_o.shape[0], max_episode_len-1, 1, 1).to(self.device)
        decoder_target = torch.cat([batch_obs_hat, agent_id_one_hot], dim=-1)   # (batch_size, max_len, N, obs_dim+N)
        mask = batch_active[:,1:].unsqueeze(-1).repeat(1, 1, 1, self.obs_dim+self.N)
        loss = (((decoder_output - decoder_target) * mask)**2).sum()/mask.sum()
                
        self.encoder_decoder_optimizer.zero_grad()
        loss.backward()
        self.encoder_decoder_optimizer.step()
        return loss
    
    def pretrain_recl(self, replay_buffer):
        batch = replay_buffer.get_training_data()
        max_episode_len = replay_buffer.max_episode_len
        batch_o = batch['obs_n'].to(self.device)
        batch_active = batch['active'].to(self.device)
        recl_loss = self.update_recl(batch_o, batch_active, max_episode_len)
        self.soft_update_params(self.actor.embedding_net.role_embedding_net.encoder, self.actor.embedding_net.role_embedding_net.target_encoder, self.tau)
        return recl_loss
    
    def update_ppo(self, max_episode_len, batch_obs, batch_s, batch_r, batch_v_n, batch_dw, batch_active, batch_avail_a_n, batch_a_n, batch_a_logprob_n):
        adv = []
        gae = 0
        with torch.no_grad(): # adv and v_target have no gradient
            # deltas.shape = (batch, max_episode_len, N)
            deltas = batch_r + self.gamma * (1-batch_dw) * batch_v_n[:, 1:] - batch_v_n[:, :-1]
            for t in reversed(range(max_episode_len)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae
                adv.insert(0, gae)
            adv = torch.stack(adv, dim=1)  # adv.shape=(batch, max_len, N)
            v_target = adv + batch_v_n[:, :-1]  # v_target.shape=(batch, max_len, N)

            if self.use_adv_norm:
                adv_copy = copy.deepcopy(adv.to('cpu').numpy())
                adv_copy[batch_active.to('cpu').numpy() == 0] = np.nan
                adv = ((adv - np.nanmean(adv_copy)) / (np.nanstd(adv_copy) + 1e-5))
                adv = adv.to(self.device)
        sum_actor_loss,  sum_critic_loss = 0, 0
        for _ in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                # rnn net need to stack according to the time
                self.actor.embedding_net.agent_embedding_net.rnn_hidden = None
                self.critic.state_gru_hidden = None
                self.critic.obs_gru_hidden = None
                agent_embeddings, tau_obs, tau_state  = [], [], []
                for t in range(max_episode_len):
                    # batch_s.shape=(batch, max_len, state_dim)
                    obs = batch_obs[index, t].reshape(-1, self.obs_dim) # (batch*N, obs_dim)
                    s = batch_s[index, t].reshape(-1, self.state_dim)   #(batch, state_dim)
                    agent_embed = self.actor.embedding_net.agent_embed_forward(obs, detach=False) # (batch*N, agent_embed_dim)
                    # h_obs = self.critic.obs_forward(obs,s.unsqueeze(1).repeat(1,self.N,1).reshape(-1, self.state_dim))      # (batch*N, rnn_dim)
                    h_obs = self.critic.obs_forward(obs)      # (batch*N, rnn_dim)
                    h_state = self.critic.state_forward(s)    # (batch, N*rnn_dim)
                    agent_embeddings.append(agent_embed.reshape(self.mini_batch_size, self.N, -1))
                    tau_obs.append(h_obs.reshape(self.mini_batch_size, self.N, -1))
                    tau_state.append(h_state.reshape(self.mini_batch_size, -1))
                # stack according to the time
                agent_embeddings = torch.stack(agent_embeddings, dim=1) # (batch, max_len, N, agent_embed_dim)
                tau_obs = torch.stack(tau_obs, dim=1)       # (batch, max_len, N, rnn_dim)
                tau_state = torch.stack(tau_state, dim=1)   # (batch, max_len, N*rnn_dim)

                # calculate prob, value
                role_embeddings = self.actor.embedding_net.role_embed_foward(agent_embeddings.reshape(-1, self.agent_embedding_dim))   # (batch*len*N, role_embed_dim)
                probs_now = self.actor.actor_forward(agent_embeddings.reshape(-1, self.agent_embedding_dim), 
                                                     role_embeddings, batch_avail_a_n[index].reshape(-1, self.action_dim))  # (batch*len*N, actor_dim)
                probs_now = probs_now.reshape(self.mini_batch_size, max_episode_len, self.N, -1)    # (batch, len, N, actor_dim)

                tau_state = tau_state.reshape(-1, self.N, self.rnn_hidden_dim)   # (batch*len, rnn_dim)->(batch*len, N, rnn_dim)
                att = self.critic.att_forward(tau_state, role_embeddings.reshape(-1, self.N, self.role_embedding_dim).detach())  # (batch*len, N, att_out_dim)
                values_now = self.critic.critic_forward(tau_obs.reshape(-1, tau_obs.shape[-1]), 
                                                        tau_state.unsqueeze(1).repeat(1,self.N,1,1).reshape(-1, self.N*self.rnn_hidden_dim),
                                                        att.unsqueeze(1).repeat(1,self.N,1,1).reshape(-1, self.N*att.shape[-1])) # (batch*len*N, 1)
                values_now = values_now.reshape(self.mini_batch_size, max_episode_len, self.N)

                # calcute loss
                dist_now = Categorical(probs_now)
                dist_entropy = dist_now.entropy()   # shape=(mini_batch, max_len, N)
                a_logprob_n_now = dist_now.log_prob(batch_a_n[index])   # shape=(mini_batch, max_len, N)
                # a/b = exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_n_now-batch_a_logprob_n[index].detach())   # ratios.shape=(mini_batch, max_len, N)
                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                actor_loss = (actor_loss * batch_active[index]).sum() / batch_active[index].sum()
                # sum_actor_loss += actor_loss.item()
                # self.actor_optimizer.zero_grad()
                # actor_loss.backward()
                # if self.use_grad_clip:
                #     torch.nn.utils.clip_grad_norm_(self.actor_parameters, 10.0)
                # self.actor_optimizer.step()

                critic_loss = (values_now - v_target[index]) ** 2
                critic_loss = (critic_loss * batch_active[index]).sum() / batch_active[index].sum()
                # sum_critic_loss += critic_loss.item()
                # self.critic_optimizer.zero_grad()
                # critic_loss.backward()
                # if self.use_grad_clip:
                #     torch.nn.utils.clip_grad_norm_(self.critic_parameters, 10.0)
                # self.critic_optimizer.step()
                
                self.ac_optimizer.zero_grad()
                ac_loss = actor_loss + critic_loss
                ac_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10.0)
                self.ac_optimizer.step()
        if self.use_lr_decay:
            self.ac_lr_decay.step()

        # if self.use_lr_decay:
        #     self.actor_lr_decay.step()
        #     self.critic_lr_decay.step()

        return sum_actor_loss, sum_critic_loss
    
                
    def update_recl(self, batch_obs, batch_active, max_episode_len):  # role embedding contrative learning
        loss = 0.0
        self.actor.embedding_net.agent_embedding_net.rnn_hidden = None
        labels = np.zeros((batch_obs.shape[0], self.N))    # (batch, N)
        for t in range(max_episode_len):    # t = 0, 1, 2...(max_episode_len-1)
            with torch.no_grad():
                agent_embedding = self.actor.embedding_net.agent_embed_forward(batch_obs[:, t].reshape(-1, self.obs_dim), detach=True)    # (batch*N, obs_dim)
            role_embedding_query = self.actor.embedding_net.role_embed_foward(agent_embedding, detach=False, ema=False).reshape(-1, self.N, self.role_embedding_dim)  # (batch, N, role_dim)
            role_embedding_key = self.actor.embedding_net.role_embed_foward(agent_embedding, detach=True, ema=True).reshape(-1, self.N, self.role_embedding_dim)

            logits = torch.bmm(role_embedding_query, self.actor.embedding_net.W.squeeze(0).expand((role_embedding_query.shape[0],self.role_embedding_dim,self.role_embedding_dim)))
            logits = torch.bmm(logits, role_embedding_key.transpose(1,2))   # (batch_size, N, N)
            logits = logits - torch.max(logits, dim=-1)[0][:,:,None]
            exp_logits = torch.exp(logits) # (batch_size, N, 1)
            agent_embedding = agent_embedding.reshape(batch_obs.shape[0],self.N, -1).to('cpu')  # shape=(batch_size,N, agent_embed_dim)
            for idx in range(agent_embedding.shape[0]): # idx = 0,1,2...(batch_size-1)
                if torch.sum(batch_active[idx, t]).item() > (self.N -1):
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
        loss /= (self.batch_size * max_episode_len * self.N*10)
        if torch.sum(batch_active[idx, t]).item() > (self.N -1):
            self.cl_optimizer.zero_grad()
            loss.backward()
            self.cl_optimizer.step()
        return loss

    def get_inputs(self, batch):
        batch_obs = batch['obs_n'].to(self.device)    # (batch, max_len, N, obs_dim)
        batch_s = batch['s'].to(self.device)    # (batch, max_len, state_dim)

        batch_r = batch['r'].to(self.device)        # (batch, max_len, N)
        batch_v_n = batch['v_n'].to(self.device)    # (batch, max_len+1, N)
        batch_dw = batch['dw'].to(self.device)      # (batch, max_len, N)
        batch_active = batch['active'].to(self.device)  # (batch, max_len, N)
        batch_avail_a_n = batch['avail_a_n']    # (batch, max_len, N, action_dim)
        batch_a_n = batch['a_n'].to(self.device)    # (batch, max_len, N)
        batch_a_logprob_n = batch['a_logprob_n'].to(self.device)    # (batch, max_len, N)
        
        return batch_obs, batch_s, batch_r, batch_v_n, batch_dw, batch_active, batch_avail_a_n, batch_a_n, batch_a_logprob_n
    
    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        



 


