import torch
import numpy as np
from smac.env import StarCraft2Env
from algorithm.mappo import MAPPO
from algorithm.acorm import ACORM
from util.replay_buffer import ReplayBuffer
import seaborn as sns
import matplotlib.pyplot as plt


class Runner(object):
    def __init__(self, args):
        self.args = args
        self.env_name = self.args.env_name
        self.seed = self.args.seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Create env
        self.env = StarCraft2Env(map_name=self.env_name, seed=self.seed)
        self.env_info = self.env.get_env_info()
        self.args.N = self.env_info["n_agents"]  # The number of agents
        self.args.obs_dim = self.env_info["obs_shape"]  # The dimensions of an agent's observation space
        self.args.state_dim = self.env_info["state_shape"]  # The dimensions of global state space
        self.args.action_dim = self.env_info["n_actions"]  # The dimensions of an agent's action space
        self.args.episode_limit = self.env_info["episode_limit"]  # Maximum number of steps per episode
        print("number of agents={}".format(self.args.N))
        print("obs_dim={}".format(self.args.obs_dim))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        self.save_path =  args.save_path + args.algorithm +'/'+ args.env_name + '/'

        # create N agent
        if args.algorithm == 'mappo':
            self.agent_n = MAPPO(self.args)
        elif args.algorithm == 'acorm':
            self.agent_n = ACORM(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        self.win_rates = []  # Record the win rates
        self.evaluate_reward = []
        self.total_steps = 0
        self.agent_embed_pretrain_epoch, self.recl_pretrain_epoch = 0, 0
        self.pretrain_agent_embed_loss, self.pretrain_recl_loss = [], []

    def run(self, ):
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1
            
            _, _, episode_steps = self.run_episode_smac(evaluate=False)  # Run an episode

            if self.agent_embed_pretrain_epoch < self.args.agent_embed_pretrain_epochs: # agent_embed_pretrain mode
                if self.replay_buffer.episode_num == self.args.batch_size:
                    self.agent_embed_pretrain_epoch += 1
                    for _  in range(1):
                        agent_embedding_loss = self.agent_n.pretrain_agent_embedding(self.replay_buffer)
                        self.pretrain_agent_embed_loss.append(agent_embedding_loss.item())
                        recl_loss = self.agent_n.pretrain_recl(self.replay_buffer)
                        self.pretrain_recl_loss.append(recl_loss.item())
                    self.replay_buffer.reset_buffer()

                    if self.agent_embed_pretrain_epoch >= self.args.agent_embed_pretrain_epochs:  # plot loss 
                        sns.set_style('whitegrid')
                        plt.figure()
                        x_step = np.array(range(len(self.pretrain_agent_embed_loss)))
                        ax = sns.lineplot(x=x_step, y=np.array(self.pretrain_agent_embed_loss).flatten(), label='agent_embedding_loss')
                        plt.ylabel('loss', fontsize=14)
                        plt.xlabel(f'step', fontsize=14)
                        plt.title(f'agent_embedding network pretrain')
                        plt.savefig(f'{self.save_path}/{self.env_name}_agent_loss_seed{self.seed}.jpg')
                        plt.figure()
                        x_step = np.array(range(len(self.pretrain_recl_loss)))
                        ax = sns.lineplot(x=x_step, y=np.array(self.pretrain_recl_loss).flatten(), label='recl_loss')
                        plt.ylabel('loss', fontsize=14)
                        plt.xlabel(f'step', fontsize=14)
                        plt.title(f'RECL network pretrain')
                        plt.savefig(f'{self.save_path}/{self.env_name}_recl_loss_seed{self.seed}.jpg')
                        print("pretrain_end!")
        
            # else:
            #     if self.recl_pretrain_epoch < self.args.recl_pretrain_epochs:   # recl_pretrain mode
            #         self.recl_pretrain_epoch += 1
            #         for _  in range(self.args.K_epochs):
            #             recl_loss = self.agent_n.pretrain_recl(self.replay_buffer)
            #             self.pretrain_recl_loss.append(recl_loss.item())
            #         self.replay_buffer.reset_buffer()
            #         if self.recl_pretrain_epoch >= self.args.recl_pretrain_epochs:  # plot loss 
            #             sns.set_style('whitegrid')
            #             plt.figure()
            #             x_step = np.array(range(len(self.pretrain_agent_embed_loss)))
            #             ax = sns.lineplot(x=x_step, y=np.array(self.pretrain_agent_embed_loss).flatten(), label='agent_embedding_loss')
            #             plt.ylabel('loss', fontsize=14)
            #             plt.xlabel(f'step', fontsize=14)
            #             plt.title(f'agent_embedding network pretrain')
            #             plt.savefig(f'{self.save_path}/{self.env_name}_agent_loss_seed{self.seed}.jpg')
            #             plt.figure()
            #             x_step = np.array(range(len(self.pretrain_recl_loss)))
            #             ax = sns.lineplot(x=x_step, y=np.array(self.pretrain_recl_loss).flatten(), label='recl_loss')
            #             plt.ylabel('loss', fontsize=14)
            #             plt.xlabel(f'step', fontsize=14)
            #             plt.title(f'RECL network pretrain')
            #             plt.savefig(f'{self.save_path}/{self.env_name}_recl_loss_seed{self.seed}.jpg')
            #             print("pretrain_end!")
            else:
                self.total_steps += episode_steps
                if self.replay_buffer.episode_num == self.args.batch_size:
                    actor_loss, critic_loss = self.agent_n.train(self.replay_buffer)
                    self.replay_buffer.reset_buffer()

        self.evaluate_policy()
        self.env.close()

    def evaluate_policy(self, ):
        win_times = 0
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            win_tag, episode_reward, _ = self.run_episode_smac(evaluate=True)
            if win_tag:
                win_times += 1
            evaluate_reward += episode_reward
        win_rate = win_times / self.args.evaluate_times
        self.win_rates.append(win_rate)
        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.evaluate_reward.append(evaluate_reward)
        print("total_steps:{} \t win_rate:{} \t evaluate_reward:{}".format(self.total_steps, win_rate, evaluate_reward))
        # # plot curve
        sns.set_style('whitegrid')
        plt.figure()
        x_step = np.array(range(len(self.win_rates)))
        ax = sns.lineplot(x=x_step, y=np.array(self.win_rates).flatten(), label=self.args.algorithm)
        plt.ylabel('win_rates', fontsize=14)
        plt.xlabel(f'step*{self.args.evaluate_freq}', fontsize=14)
        plt.title(f'{self.args.algorithm} on {self.env_name}')
        plt.savefig(f'{self.save_path}/{self.env_name}_seed{self.seed}.jpg')

        # Save the win rates
        np.save(f'{self.save_path}/{self.env_name}_seed{self.seed}.npy', np.array(self.win_rates))
        np.save(f'{self.save_path}/{self.env_name}_seed{self.seed}_return.npy', np.array(self.evaluate_reward))
        
    def run_episode_smac(self, evaluate=False):
        win_tag = False
        episode_reward = 0
        self.env.reset()
        self.agent_n.actor.embedding_net.agent_embedding_net.rnn_hidden = None
        self.agent_n.critic.state_gru_hidden = None
        self.agent_n.critic.obs_gru_hidden = None
        for episode_step in range(self.args.episode_limit):
            obs_n = self.env.get_obs()  # obs_n.shape=(N, obs_dim)
            temp_obs_n = obs_n
            s = self.env.get_state()    # s.shape=(state_dim,)
            avail_a_n = self.env.get_avail_actions()    # avail_a_n

            temp_obs_n = torch.tensor(np.array(temp_obs_n),dtype=torch.float32).to(self.device)
            agent_embedding = self.agent_n.actor.embedding_net.agent_embed_forward(temp_obs_n, detach=True)  # (N, agent_embed_dim)
            role_embedding = self.agent_n.actor.embedding_net.role_embed_foward(agent_embedding, detach=True, ema=False)   # (N, role_embed_dim)
            
            a_n, a_logprob_n = self.agent_n.choose_action(agent_embedding, role_embedding, avail_a_n, evaluate=evaluate)
            r, done, info = self.env.step(a_n)
            win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
            episode_reward += r

            if not evaluate:
                if done and episode_step + 1 != self.args.episode_limit:
                    dw = True
                else: 
                    dw = False
                v_n = self.agent_n.get_value(s, temp_obs_n, role_embedding.unsqueeze(0))  # Get the state values (V(s)) of N agents
                # Store the transition
                self.replay_buffer.store_transition(episode_step, obs_n, s, v_n, avail_a_n, a_n, a_logprob_n, r, dw)

            if done:
                break

        if not evaluate:
            # An episode is over, store obs_n, s and avail_a_n in the last step
            obs_n = self.env.get_obs()
            s = self.env.get_state()

            obs_n = torch.tensor(np.array(obs_n),dtype=torch.float32).to(self.device)
            agent_embedding = self.agent_n.actor.embedding_net.agent_embed_forward(obs_n, detach=True)  # (N, agent_embed_dim)
            role_embedding = self.agent_n.actor.embedding_net.role_embed_foward(agent_embedding, detach=True, ema=False)   # (N, role_embed_dim)
            v_n = self.agent_n.get_value(s, obs_n, role_embedding.unsqueeze(0))
            self.replay_buffer.store_last_value(episode_step+1, v_n)

        return win_tag, episode_reward, episode_step+1