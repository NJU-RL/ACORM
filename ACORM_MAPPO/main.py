from run import Runner
import argparse
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in SMAC environment")
    parser.add_argument("--algorithm", type=str, default="acorm", help="acorm or mappo")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=32, help="Evaluate times")
    parser.add_argument("--save_freq", type=int, default=int(1e5), help="Save frequency")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The dimension of the hidden layer of RNN")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--actor_lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--critic_lr", type=float, default=8e-4, help="Learning rate")
    parser.add_argument("--lr_decay_steps", type=int, default=500, help="every steps decay steps")
    parser.add_argument("--lr_decay_rate", type=float, default=0.98, help="learn decay rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=5, help="GAE parameter")
    parser.add_argument("--entropy_coef", type=float, default=0.015, help="policy entropy")
    
    # ppo tricks
    parser.add_argument("--use_lr_decay", type=bool, default=False, help="Trick:learning rate Decay")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick:advantage normalization")
    parser.add_argument("--use_grad_clip", type=bool, default=False, help="Trick: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick: orthogonal initialization")
    parser.add_argument("--add_agent_id", type=bool, default=True, help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--use_agent_specific", type=bool, default=True, help="Whether to use agent specific global state.")

    parser.add_argument('--env_name', type=str, default='MMM2')  #['3m', '8m', '2s3z']
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--seed", type=int, default=123, help="random seed")

    # recl
    parser.add_argument("--agent_embedding_dim", type=int, default=64, help="The dimension of the agent embedding")
    parser.add_argument("--role_embedding_dim", type=int, default=32, help="The dimension of the role embedding")
    parser.add_argument("--cluster_num", type=int, default=int(3), help="the cluster number of k-means")
    parser.add_argument("--cl_lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--agent_embedding_lr", type=float, default=1e-3, help="agent_embedding Learning rate")
    parser.add_argument("--train_recl_freq", type=int, default=2, help="Train frequency of the contrastive role embedding")
    parser.add_argument("--multi_steps", type=int, default=1, help="Train frequency of the RECL network")
    parser.add_argument("--tau", type=float, default=0.005, help="If use soft update")
    parser.add_argument("--agent_embed_pretrain_epochs", type=int, default=150, help="pretrain steps")
    parser.add_argument("--recl_pretrain_epochs", type=int, default=120, help="pretrain steps")
 
    # attention
    parser.add_argument("--att_dim", type=int, default=256, help="The dimension of the attention net")
    parser.add_argument("--att_out_dim", type=int, default=64, help="The dimension of the attention net")
    parser.add_argument("--n_heads", type=int, default=8, help="multi-head attention")
    parser.add_argument("--soft_temp", type=float, default=1.0, help="soft tempture")

    # save path
    parser.add_argument('--save_path', type=str, default='./result/')
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')
    runner = Runner(args)
    runner.run()


