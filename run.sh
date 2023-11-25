nohup python main.py  --algorithm ACORM    \
                      --env_name 'MMM2'  \
                      --device 'cuda:1'  \
                      --max_train_steps 3000000 \
                      --seed 4  \
                      --epsilon 1.0                 \
                      --epsilon_decay_steps 80000 \
                      --epsilon_min 0.02     \
                      --use_hard_update False \
                      --use_lr_decay True     \
                      --lr_decay_steps 500    \
                      --lr_decay_rate 0.98    \
                      --train_recl_freq 100 \
                      --use_ln False \
                      --role_tau 0.005  \
                      --cluster_num 3   \
                      --agent_embedding_dim 128 \
                      --hyper_layers_num 2      \
                      --lr +6e-4                \
                      --recl_lr +8e-4           \
                      --role_embedding_dim 64   \
                      --save_path './result/acorm'\
                      --att_dim 128  \
                      --att_out_dim 64  \
                      --n_heads 4   \
                      --soft_temperature 1.0    \
                      --state_embed_dim 64      \
                      &


