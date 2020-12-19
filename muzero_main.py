#!/usr/bin/env python
# coding: utf-8

# In[1]:



import gym
import ray
from games.utils import make_pole_config
from muzero.train import muzero
import torch

# In[2]:



# In[3]:





ray.init()

env_name = "CartPole-v1"
muzero_config = make_pole_config(env_name)
model = muzero(muzero_config)
ray.shutdown()
torch.save(model, "./model_prova_3")

