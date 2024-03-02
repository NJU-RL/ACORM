import torch
from util.net import Actor, Critic
from torch.distributions import Categorical
from torch.utils.data.sampler import *
import numpy as np
import copy
from torch.optim.lr_scheduler import StepLR

class MAPPO(object):
    def __init__(self, args):
        self.N = args.N
        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim

        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps

        self.lr = args.lr
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.clip_epsilon = args.clip_epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef

        self.use_lr_decay = args.use_lr_decay
        self.lr_decay_steps = args.lr_decay_steps
        self.lr_decay_rate = args.lr_decay_rate
        self.use_adv_norm = args.use_adv_norm
        self.use_grad_clip = args.use_grad_clip
        self.add_agent_id = args.add_agent_id
        self.use_agent_specific = args.use_agent_specific

        self.actor_input_dim = args.obs_dim
        self.critic_input_dim = args.state_dim

        if self.add_agent_id:
            self.actor_input_dim += args.N
            self.critic_input_dim += args.N
        if self.use_agent_specific:
            self.critic_input_dim += args.obs_dim
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        self.actor = Actor(args, self.actor_input_dim)
        self.critic = Critic(args, self.critic_input_dim)

        self.ac_parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr, eps=1e-5)
        self.ac_lr_decay = StepLR(self.ac_optimizer, step_size=self.lr_decay_steps, gamma=self.lr_decay_rate)

        self.actor.to(self.device)
        self.critic.to(self.device)
          
    def choose_action(self, obs_n, avail_a_n, evaluate):
        with torch.no_grad():
            actor_input = torch.tensor(np.array(obs_n), dtype=torch.float32)  # obs_n.shape=(N, obs_dim)
            if self.add_agent_id:
                actor_input = torch.cat([actor_input, torch.eye(self.N)], dim=-1)   # input.shape=(N, obs_dim+N)
            avail_a_n = torch.tensor(avail_a_n, dtype=torch.float32)  # avail_a_n.shape=(N, action_dim)
            actor_input = actor_input.to(self.device)
            avail_a_n = avail_a_n.to(self.device)
            prob = self.actor(actor_input, avail_a_n)   # prob.shape=(N, action_dim)

            if evaluate:
                a_n = prob.argmax(dim=-1).to('cpu')
                return a_n.numpy(), None
            else:
                dist = Categorical(probs=prob)
                a_n = dist.sample()
                a_logprob_n = dist.log_prob(a_n)
                return a_n.to('cpu').numpy(), a_logprob_n.to('cpu').numpy()
            
    def get_value(self, s, obs_n):
        with torch.no_grad():
            obs_n = torch.tensor(np.array(obs_n),dtype=torch.float32)
            critic_input = torch.tensor(np.array(s), dtype=torch.float32).unsqueeze(0).repeat(self.N,1)   # (state_dim,)->(N, state_dim)
            if self.use_agent_specific:
                critic_input = torch.cat([critic_input, obs_n], dim=-1) # (N, state_dim+obs_dim)
            if self.add_agent_id:
                critic_input = torch.cat([critic_input, torch.eye(self.N)], dim=-1) # (N, input_dim)
            critic_input = critic_input.to(self.device)
            v_n = self.critic(critic_input) # v_n.shape=(N, 1)
            return v_n.to('cpu').numpy().flatten()
        
    def train(self, replay_buffer):
        batch = replay_buffer.get_training_data()
        max_episode_len = replay_buffer.max_episode_len
        actor_inputs, critic_inputs, batch_r, batch_v_n, batch_dw, batch_active, batch_avail_a_n, batch_a_n, batch_a_logprob_n = self.get_inputs(batch)

        # Calculate the advantage using GAE
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
            # normalization
            if self.use_adv_norm:
                adv_copy = copy.deepcopy(adv.to('cpu').numpy())
                adv_copy[batch['active'].numpy() == 0] = np.nan
                adv = ((adv - np.nanmean(adv_copy)) / (np.nanstd(adv_copy) + 1e-5))
                adv = adv.to(self.device) 

        for _ in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                # probs_now.shape=(mini_batch, max_len, N, actor_dim)
                # values_now.shape=(mini_batch, max_len, N)
                self.actor.rnn_hidden = None
                self.critic.rnn_hidden = None
                probs_now, values_now = [], []
                for t in range(max_episode_len):
                    prob = self.actor(actor_inputs[index, t].reshape(self.mini_batch_size*self.N, -1),
                                      batch_avail_a_n[index, t].reshape(self.mini_batch_size*self.N, -1))   # prob.shape=(mini_batch*N,action_dim）
                    probs_now.append(prob.reshape(self.mini_batch_size, self.N, -1))
                    value = self.critic(critic_inputs[index, t].reshape(self.mini_batch_size*self.N, -1))   # value.shape(mini_batch*N, 1)
                    values_now.append(value.reshape(self.mini_batch_size, self.N))
                # stack according to the time
                probs_now = torch.stack(probs_now, dim=1)
                values_now = torch.stack(values_now, dim=1)

                dist_now = Categorical(probs_now)
                dist_entropy = dist_now.entropy()   # shape=(mini_batch, max_len, N)
                a_logprob_n_now = dist_now.log_prob(batch_a_n[index])   # shape=(mini_batch, max_len, N)
                # a/b = exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_n_now-batch_a_logprob_n[index].detach())   # ratios.shape=(mini_batch, max_len, N)
                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                actor_loss = (actor_loss * batch_active[index]).sum() / batch_active[index].sum()

                critic_loss = (values_now - v_target[index]) ** 2
                critic_loss = (critic_loss * batch_active[index]).sum() / batch_active[index].sum()

                self.ac_optimizer.zero_grad()
                ac_loss = actor_loss + critic_loss
                ac_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10.0)
                self.ac_optimizer.step()
        if self.use_lr_decay:
            self.ac_lr_decay.step()

    def get_inputs(self, batch):
        # batch['obs_n'].shape=(batch, max_len, N, obs_dim)
        # batch['s].shape=(batch, max_len, state_dim)
        actor_inputs = copy.deepcopy(batch['obs_n'])
        critic_inputs = copy.deepcopy(batch['s'].unsqueeze(2).repeat(1, 1, self.N, 1))
        if self.use_agent_specific:
            critic_inputs = torch.cat([critic_inputs, batch['obs_n']], dim=-1)  # 
        if self.add_agent_id:
            agent_id_one_hot = torch.eye(self.N).unsqueeze(0).unsqueeze(0).repeat(self.batch_size, batch['s'].shape[1], 1, 1)
            actor_inputs = torch.cat([actor_inputs, agent_id_one_hot], dim=-1)      # shape=(batch, max_len, N, obs_dim+N)
            critic_inputs = torch.cat([critic_inputs, agent_id_one_hot], dim=-1)    # shape=(batch, max_len, N, state_dim+obs_dim+N)
            
        actor_inputs = actor_inputs.to(self.device)
        critic_inputs = critic_inputs.to(self.device)
        batch_r = batch['r'].to(self.device)        # (batch, max_len, N)
        batch_v_n = batch['v_n'].to(self.device)    # (batch, max_len+1, N)
        batch_dw = batch['dw'].to(self.device)      # (batch, max_len, N)
        batch_active = batch['active'].to(self.device)  # (batch, max_len, N)
        batch_avail_a_n = batch['avail_a_n']    # (batch, max_len, N, action_dim)
        batch_a_n = batch['a_n'].to(self.device)    # (batch, max_len, N, action_dim)
        batch_a_logprob_n = batch['a_logprob_n'].to(self.device)
        return actor_inputs, critic_inputs, batch_r, batch_v_n, batch_dw, batch_active, batch_avail_a_n, batch_a_n, batch_a_logprob_n
    
    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

