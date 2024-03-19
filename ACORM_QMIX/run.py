import torch
import numpy as np
from algorithm.vdn_qmix import VDN_QMIX
from algorithm.acorm import ACORM_Agent
from util.replay_buffer import ReplayBuffer
from smac.env import StarCraft2Env
import seaborn as sns
import matplotlib.pyplot as plt
import time

class Runner:
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
        self.save_path =  args.save_path
        self.model_path = args.model_path

        # Create N agents
        if args.algorithm in ['QMIX', 'VDN']:
            self.agent_n = VDN_QMIX(self.args)
        elif args.algorithm == 'ACORM':
            self.agent_n = ACORM_Agent(self.args)
        self.replay_buffer = ReplayBuffer(self.args, self.args.buffer_size)

        # # Create a tensorboard
        # self.writer = SummaryWriter(log_dir='./runs/{}/{}_env_{}_number_{}_seed_{}'.format(self.args.algorithm, self.args.algorithm, self.env_name, self.number, self.seed))

        self.epsilon = self.args.epsilon  # Initialize the epsilon
        self.win_rates = []  # Record the win rates
        self.evaluate_reward = []
        self.total_steps = 0
        self.agent_embed_pretrain_epoch, self.recl_pretrain_epoch = 0, 0
        self.pretrain_agent_embed_loss, self.pretrain_recl_loss = [], []
        self.args.agent_embed_pretrain_epochs =0
        self.args.recl_pretrain_epochs = 0

    def run(self, ):
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            _, _, episode_steps = self.run_episode_smac(evaluate=False)  # Run an episode
            
            if self.agent_embed_pretrain_epoch < self.args.agent_embed_pretrain_epochs:
                if self.replay_buffer.current_size >= self.args.batch_size:
                    self.agent_embed_pretrain_epoch += 1
                    agent_embedding_loss = self.agent_n.pretrain_agent_embedding(self.replay_buffer)
                    self.pretrain_agent_embed_loss.append(agent_embedding_loss.item())
            else:
                if self.recl_pretrain_epoch < self.args.recl_pretrain_epochs:
                    self.recl_pretrain_epoch += 1
                    recl_loss = self.agent_n.pretrain_recl(self.replay_buffer)
                    self.pretrain_recl_loss.append(recl_loss.item())
                    
                else:                                                          
                    self.total_steps += episode_steps
                    if self.replay_buffer.current_size >= self.args.batch_size:
                        self.agent_n.train(self.replay_buffer)  # Training
                    
        self.evaluate_policy()
         # save model
        model_path = f'{self.model_path}/{self.env_name}_seed{self.seed}_'
        torch.save(self.agent_n.eval_Q_net, model_path + 'q_net.pth')
        torch.save(self.agent_n.RECL.role_embedding_net, model_path + 'role_net.pth')
        torch.save(self.agent_n.RECL.agent_embedding_net, model_path+'agent_embed_net.pth')
        torch.save(self.agent_n.eval_mix_net.attention_net, model_path+'attention_net.pth')
        torch.save(self.agent_n.eval_mix_net, model_path+'mix_net.pth')
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
        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.win_rates.append(win_rate)
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
        
        self.agent_n.eval_Q_net.rnn_hidden = None
        if self.args.algorithm == 'ACORM':
            self.agent_n.RECL.agent_embedding_net.rnn_hidden = None

        last_onehot_a_n = np.zeros((self.args.N, self.args.action_dim))  # Last actions of N agents(one-hot)
        for episode_step in range(self.args.episode_limit):
            obs_n = self.env.get_obs()  # obs_n.shape=(N,obs_dim)
            s = self.env.get_state()  # s.shape=(state_dim,)
            avail_a_n = self.env.get_avail_actions()  # Get available actions of N agents, avail_a_n.shape=(N,action_dim)
            epsilon = 0 if evaluate else self.epsilon
            
            if self.args.algorithm == 'ACORM':
                role_embedding = self.agent_n.get_role_embedding(obs_n, last_onehot_a_n)
                a_n = self.agent_n.choose_action(obs_n, last_onehot_a_n, role_embedding, avail_a_n, epsilon)
            else:
                a_n = self.agent_n.choose_action(obs_n, last_onehot_a_n, avail_a_n, epsilon)

            r, done, info = self.env.step(a_n)  # Take a step
            win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
            episode_reward += r

            if not evaluate:
                """"
                    When dead or win or reaching the episode_limit, done will be Ture, we need to distinguish them;
                    dw means dead or win,there is no next state s';
                    but when reaching the max_episode_steps,there is a next state s' actually.
                """
                if done and episode_step + 1 != self.args.episode_limit:
                    dw = True
                else:
                    dw = False

                # Store the transition
                self.replay_buffer.store_transition(episode_step, obs_n, s, avail_a_n, last_onehot_a_n, a_n, r, dw)
                last_onehot_a_n = np.eye(self.args.action_dim)[a_n]  # Convert actions to one-hot vectors
                # obs_a_n_buffer[episode_step] = obs_n
                # Decay the epsilon
                self.epsilon = self.epsilon - self.args.epsilon_decay if self.epsilon - self.args.epsilon_decay > self.args.epsilon_min else self.args.epsilon_min

            if done:
                break

        if not evaluate:
            # An episode is over, store obs_n, s and avail_a_n in the last step
            obs_n = self.env.get_obs()
            s = self.env.get_state()
            avail_a_n = self.env.get_avail_actions()
            self.replay_buffer.store_last_step(episode_step + 1, obs_n, s, avail_a_n, last_onehot_a_n)
        return win_tag, episode_reward, episode_step+1