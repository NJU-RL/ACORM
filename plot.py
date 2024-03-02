import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def smooth(y, radius, mode='two_sided', valid_only=False):
    assert mode in ('two_sided', 'causal')
    if len(y) < 2*radius+1:
        return np.ones_like(y) * y.mean()
    elif mode == 'two_sided':
        convkernel = np.ones(2 * radius+1)
        out = np.convolve(y, convkernel,mode='same') / np.convolve(np.ones_like(y), convkernel, mode='same')
        if valid_only:
            out[:radius] = out[-radius:] = np.nan
    elif mode == 'causal':
        convkernel = np.ones(radius)
        out = np.convolve(y, convkernel,mode='full') / np.convolve(np.ones_like(y), convkernel, mode='full')
        out = out[:-radius+1]
        if valid_only:
            out[:radius] = np.nan
    return out


parser = argparse.ArgumentParser("Hyperparameter Setting for QMIX,VDN and RECL_QMIX in SMAC environment")
parser.add_argument("--algorithm", default='ACORM_QMIX', help="env_names")
args = parser.parse_args()

if args.algorithm == 'ACORM_QMIX':
    algs = ['acorm','qmix']
    labels = ['ACORM(Ours)','QMIX']
elif args.algorithm == 'ACORM_MAPPO':
    algs = ['acorm','mappo']
    labels = ['ACORM(Ours)','MAPPO']
color = ['#D62728', '#2A9F2A', '#FF7F0E','#1B75B4','#9467BD','#7F7F7F','#E377C2', '#BBBC1F']

env_names = ['2s3z', '1c3s5z', '3s5z',  # easy
             'bane_vs_bane','2c_vs_64zg','5m_vs_6m','10m_vs_11m',   # hard
             '27m_vs_30m','MMM2','3s5z_vs_3s6z','6h_vs_8z','corridor']  # super hard
env_difficulty = ['easy']*3 + ['hard']*4 + ['super hard']*5

sns.set_style('ticks')
sns.set_context('talk')
fig = plt.figure(figsize=(20, 18), dpi=800)
# Grid = plt.GridSpec(2, 3, wspace=0.2, hspace=0.4)
Grid = plt.GridSpec(4, 3, wspace=0.2, hspace=0.4)
plt.rcParams.update({'font.size': 15})


for sub_i, env_name in enumerate(env_names):
    print(env_name)
    # fig=plt.figure(figsize=(6, 4),dpi=120)
    sub_ax = plt.subplot(Grid[sub_i//3,sub_i%3])
    plt.title(f'{env_name} ({env_difficulty[sub_i]})')
    for index, alg in enumerate(algs):
        print(alg)
        vdn_qmix_data = []
        for seed in range(1):
            dir = (f'./{args.algorithm}/result/sacred/{alg}/{env_name}_seed{seed}.npy')
            # if env_name in ['corridor']:
            #     data = np.load(dir, allow_pickle=True)[:980]
            if env_name in ['3s5z_vs_3s6z','6h_vs_8z','corridor']:
                data = np.load(dir, allow_pickle=True)[:600]
            elif env_name in ['bane_vs_bane','2s3z','1c3s5z']:
                data = np.load(dir, allow_pickle=True)[:200]
            elif env_name in ['27m_vs_30m','10m_vs_11m','5m_vs_6m','2c_vs_64zg']:
                data = np.load(dir, allow_pickle=True)[:300]
            else:
                data = np.load(dir, allow_pickle=True)[:400]
            data[0:5] = 0.0
            data = smooth(data, radius=3)
            vdn_qmix_data.append(data)
        vdn_qmix_data = np.array(vdn_qmix_data)
        x_step = np.tile(np.array(range(vdn_qmix_data.shape[1]))*5000, vdn_qmix_data.shape[0])
        ax = sns.lineplot(x=x_step, y=vdn_qmix_data.flatten(),label=labels[index], color=color[index], linewidth=2)  

    plt.grid(True,linestyle='-.',alpha=0.4)
    plt.legend(fontsize = 11)
    plt.ylabel('Test Win Rate', labelpad=-0.5)
    plt.xlabel(f'Timesteps')
    if env_name in ['3s5z_vs_3s6z','6h_vs_8z', 'corridor']:
        plt.xticks(np.array(range(data.shape[0]//100 +1))*500000)
    plt.yticks(np.array(range(0, 10+2, 2))/10)
    handles, labels = sub_ax.get_legend_handles_labels()
    sub_ax.legend_.remove()

# plt.legend(handles, labels, ncol=6, bbox_to_anchor=(0.44, 5.55))
plt.legend(handles, labels, ncol=8, bbox_to_anchor=(0.82, 5.55))

plt.tight_layout()
plt.savefig(f'./smac_{args.algorithm}.jpg',bbox_inches='tight')