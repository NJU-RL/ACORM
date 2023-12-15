# **Attention-Guided Contrastive Role Representations for Multi-agent Reinforcement Learning**

Zican Hu, Zongzhang Zhang, Huaxiong Li, Chunlin Chen, Hongyu Ding, Zhi Wang

A link to our paper can be found on [arXiv](https://arxiv.org/abs/2312.04819)

## **Overview**

![ACORM_QMIX](./ACORM_QMIX.jpg)

## **Instructions**

ACORM tested on two benchmark tasks [SMAC ](https://github.com/oxwhirl/smac) and [GRF](https://github.com/google-research/football) based on two algorithm framework [QMIX](https://arxiv.org/abs/1803.11485) and [MAPPO](https://arxiv.org/abs/2103.01955).

## **Citation**

If you find our code and paper can help, please cite our paper as:
```
@article{hu2023attentionguided,
  title={Attention-Guided Contrastive Role Representations for Multi-Agent Reinforcement Learning},
  author={Zican Hu and Zongzhang Zhang and Huaxiong Li and Chunlin Chen and Hongyu Ding and Zhi Wang},
  journal={arXiv preprint arXiv:2312.04819},
  year={2023}
}
```
## **experiment instructions**

### **Installation instructions**

```python
conda create -n acorm python=3.9.16 -y
conda activate acorm
pip install -r requirements.txt
```

### Run an experiment

You can execute the following command to run ACORM with a map config, such as `MMM2`:

```python
python main.py --algorithm ACORM --env_name MMM2 --cluster_num 3 --max_train_steps 3050000
```

Or you can  can modify the parameters of the `run.sh` file, then execute shell file:

```python
sh run.sh
```

All results will be stored in the `results` folder. You can see the console output, config, and tensorboard logging in the `results/tb_logs` folder.

You can plot the curve with `seaborn`:

```python
python plot.py --env_names 'MMM2' 'corridor'
```

## License

Code licensed under the Apache License v2.0.

