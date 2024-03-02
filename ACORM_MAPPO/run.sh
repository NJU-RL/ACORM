nohup python main.py  --algorithm 'acorm'    \
                      --env_name 'MMM2'  \
                      --seed 3         \
                      --device 'cuda:1'  \
                      --max_train_steps 3000000 \
                      --actor_lr +6e-4 \
                      --critic_lr +8e-4 \
                      --cl_lr +5e-4 \
                      --batch_size 32   \
                      --mini_batch_size 32   \
                      --agent_embedding_dim 64 \
                      --role_embedding_dim 32 \
                      --rnn_hidden_dim 64   \
                      --gamma 0.99      \
                      --lamda 0.95      \
                      --clip_epsilon 0.2     \
                      --K_epochs 5     \
                      --entropy_coef 0.015   \
                      --add_agent_id False   \
                      --use_adv_norm True   \
                      --use_grad_clip False  \
                      --use_orthogonal_init False    \
                      --use_lr_decay False       \
                      --cluster_num 3   \
                      --train_recl_freq 16 \
                      --multi_steps 1   \
                      --tau 0.005   \
                      --att_dim 128  \
                      --att_out_dim 64  \
                      --n_heads 4   \
                      --soft_temp 1.0 \
                      &


