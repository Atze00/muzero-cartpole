from muzero.selfplay import play_game
from games.utils import make_pole_config
import torch
import argparse
import time
parser = argparse.ArgumentParser()
parser.add_argument('--n_evaluation', type=int, required=False,default = 1)
parser.add_argument('--visualize', action='store_true')
args = parser.parse_args()

visualize = args.visualize
env_name = "CartPole-v1"
muzero_config = make_pole_config(env_name)
network = torch.load("./model_prova_2")
games = []
rewards = []
for i in range(args.n_evaluation):    
    with torch.no_grad():
        game = play_game(muzero_config, network.eval(), render = True)
    games.append(game)
    rewards.append(sum(game.rewards))
print(sum(rewards)/len(rewards))
